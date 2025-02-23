import torch
import torch.nn as nn
from torchmetrics import Accuracy
import os
import math
from scipy.stats import entropy
from copy import deepcopy
from torch.nn.utils.weight_norm import WeightNorm

from networks import BaseModule, FeatureExtractor
from utils import SupConLoss, HScore
from augmentation import get_tta_transforms


class COMET(BaseModule):
    def __init__(self, datamodule, rejection_threshold=0.5, feature_dim=256, lr=1e-3, ckpt_dir='', cl_projection_dim=128, cl_temperature=0.1,
                 lbd=0.01, use_source_prototypes=False, loss_type='contrastive+entropy', pseudo_label_quantity=1.0, pseudo_label_quality=1.0, alpha_threshold=0.5):
        super(COMET, self).__init__(datamodule, feature_dim, lr, rejection_threshold, ckpt_dir)

        self.ckpt_dir = ckpt_dir
        self.class_prototypes = None
        self.prototype_sum = torch.zeros(self.known_classes_num, feature_dim)
        self.prototype_sample_counter = torch.zeros(self.known_classes_num, 1)

        self.total_online_tta_acc = Accuracy(task='multiclass', num_classes=self.known_classes_num + 1)
        self.total_online_tta_hscore = HScore(self.known_classes_num, datamodule.shared_class_num)

        cl_projector = nn.Sequential(nn.Linear(self.feature_extractor.feature_dim, cl_projection_dim),
                                     nn.ReLU(), nn.Linear(cl_projection_dim, cl_projection_dim)).to(self.device)
        self.contrastive_loss = SupConLoss(projector=cl_projector, temperature=cl_temperature)
        self.tta_transform = get_tta_transforms()

        self.lbd = lbd
        self.loss_type = loss_type

        self.use_source_prototypes = use_source_prototypes

        self.pseudo_label_quantity = pseudo_label_quantity
        self.pseudo_label_quality = pseudo_label_quality
        print(f"Pseudo-Label Quality: {self.pseudo_label_quality}")
        print(f"Pseudo-Label Quantity: {self.pseudo_label_quantity}")
        self.alpha_threshold = alpha_threshold
        self.selected_sample_ids           = []
        self.selected_sample_ids_correct   = []
        self.selected_sample_ids_incorrect = [] 

        self.automatic_optimization = False

    def configure_optimizers(self):
        # define different learning rates for different subnetworks
        params_group = []

        for k, v in self.backbone.named_parameters():
            params_group += [{'params': v, 'lr': self.lr * 0.1}]
        for k, v in self.feature_extractor.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]
        for k, v in self.classifier.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]
        for k, v in self.contrastive_loss.projector.named_parameters():
            params_group += [{'params': v, 'lr': self.lr}]

        optimizer = torch.optim.SGD(params_group, momentum=0.9, nesterov=True)
        return optimizer

    def on_train_start(self):
        if torch.cuda.is_available():
            self.class_prototypes = torch.load(self.ckpt_dir, map_location=torch.device('cuda'))['class_prototypes']
        else:
            self.class_prototypes = torch.load(self.ckpt_dir, map_location=torch.device('cpu'))['class_prototypes']

    def on_fit_start(self):
        print("Sample selection...")
        train_loader = self.trainer.datamodule.train_dataloader()

        if self.pseudo_label_quantity == 1.0 and self.pseudo_label_quality == 1.0:
            for x, y, sample_ids in train_loader:
                for sample_id in sample_ids:                
                    self.selected_sample_ids.append(sample_id.item())
            self.selected_sample_ids_correct = self.selected_sample_ids
        else:
            self.on_test_model_eval()

            ### Compute entropies
            y_hat_entropy_list = []
            for batch_idx, (x, y, sample_ids) in enumerate(train_loader):
                x = x.to(self.device)
                # Compute y_hat_entropy
                y_hat, features = self.forward(x, apply_softmax=True)
                y_hat_entropy   = -torch.matmul(y_hat, torch.log(y_hat.T)) / torch.log(torch.tensor(self.known_classes_num))
                y_hat_entropy   = torch.diagonal(y_hat_entropy)

                for entropy_value, sample_id in zip(y_hat_entropy, sample_ids):
                    y_hat_entropy_list.append((entropy_value.item(), sample_id.item()))

            y_hat_entropy_list.sort(key=lambda x: x[0])

            ### 1. Select samples to be used for adaptation according to given pseudo label quantity
            if self.pseudo_label_quantity == 1.0:
                for x, y, sample_ids in train_loader:
                    for sample_id in sample_ids:                
                        self.selected_sample_ids.append(sample_id.item())
            else:
                total_samples  = len(y_hat_entropy_list)
                num_samples_to_use = int(self.pseudo_label_quantity * total_samples)

                entropy_distance_list = []
                for elem in y_hat_entropy_list:
                    entropy_value = elem[0]
                    sample_id     = elem[1]
                    if entropy_value <= 0.5:
                        entropy_distance_list.append((entropy_value, entropy_value, sample_id))
                    else:
                        entropy_distance_list.append((1.0-entropy_value, entropy_value, sample_id))
                entropy_distance_list_sorted = sorted(entropy_distance_list, key=lambda x:x[0])
                resulting_samples            = entropy_distance_list_sorted[:num_samples_to_use]
                self.selected_sample_ids     = [elem[-1] for elem in resulting_samples]

            ### 2. Labeling of selected samples
            if self.pseudo_label_quality == 1.0:
                self.selected_sample_ids_correct = self.selected_sample_ids
            else:
                remaining_entropy_list = [item for item in y_hat_entropy_list if item[1] in self.selected_sample_ids]
                num_correct_samples_to_use = int(self.pseudo_label_quality * len(remaining_entropy_list))

                remaining_entropy_distance_list = []
                for elem in remaining_entropy_list:
                    entropy_value = elem[0]
                    sample_id     = elem[1]
                    if entropy_value <= 0.5:
                        remaining_entropy_distance_list.append((entropy_value, entropy_value, sample_id))
                    else:
                        remaining_entropy_distance_list.append((1.0-entropy_value, entropy_value, sample_id))
                remaining_entropy_distance_list_sorted = sorted(remaining_entropy_distance_list, key=lambda x:x[0])

                # Determine the correct and incorrect sample ids
                remaining_sample_ids_sorted        = [elem[-1] for elem in remaining_entropy_distance_list_sorted]
                self.selected_sample_ids_correct   = remaining_sample_ids_sorted[:num_correct_samples_to_use]
                self.selected_sample_ids_incorrect = remaining_sample_ids_sorted[num_correct_samples_to_use:]

    def generate_pseudo_labels(self, y_hat, y, sample_ids):
        counter_different_known = 0
        counter_unknown = 0

        batch_size    = len(y_hat)
        confident_idx = []  
        pseudo_labels = []  

        if self.pseudo_label_quantity == 1.0 and self.pseudo_label_quality == 1.0:
            confident_idx = list(range(batch_size))
            pseudo_labels = y 
        else:
            correct_indices   = [idx for idx, sample_id in enumerate(sample_ids) if sample_id.item() in self.selected_sample_ids_correct]
            incorrect_indices = [idx for idx, sample_id in enumerate(sample_ids) if sample_id.item() in self.selected_sample_ids_incorrect]
            confident_idx     = correct_indices + incorrect_indices

            for index in confident_idx:
                true_label = y[index]
                classifier_output = y_hat[index]

                if index in correct_indices:
                    pseudo_labels.append(true_label)  # Assign true label
                elif index in incorrect_indices:
                    # Assign wrong pseudo-label
                    if true_label == self.known_classes_num:
                        # If true label is "Unknown" assign most probable known class
                        most_prob_class_label = torch.argmax(classifier_output).item()
                        pseudo_labels.append(most_prob_class_label)
                    else: # If true label is one of the known classes
                        masked_probabilities = classifier_output.clone()
                        masked_probabilities[true_label] = float('-inf')
                        largest_prob, largest_index = torch.max(masked_probabilities, dim=0)

                        entropy = -torch.sum(classifier_output * torch.log(classifier_output + 1e-10)) / torch.log(torch.tensor(self.known_classes_num))
                        adaptive_threshold = self.alpha_threshold * entropy 

                        if largest_prob < adaptive_threshold:
                            # assign unknown class as wrong pseudo-label
                            pseudo_labels.append(self.known_classes_num)
                            counter_unknown += 1
                        else:
                            # assign class with largest probability besides the true class as wrong pseudo-label
                            pseudo_labels.append(largest_index.item())
                            counter_different_known += 1

        print(f"Number of samples labeled as different known class: {counter_different_known}")
        print(f"Number of samples labeled as unknown: {counter_unknown}")

        pseudo_labels = torch.tensor(pseudo_labels, device=self.device)
        confident_idx = torch.tensor(confident_idx, device=self.device)

        return confident_idx, pseudo_labels
    
    def calculate_kld(self, likelihood, true_dist):
        T = 0.1
        dividend = torch.sum(torch.exp(likelihood / T), dim=1)
        logarithmus = - torch.log(dividend)
        divisor = torch.sum(true_dist, dim=1)
        kld_values = - (1 / likelihood.shape[1]) * divisor * logarithmus
        return kld_values
    
    def calculate_cosine_similarity(self, mu, feat):
        cosine_sim = torch.nn.functional.cosine_similarity(mu.unsqueeze(0), feat.unsqueeze(1), dim=2)
        return cosine_sim

    def on_train_epoch_end(self):
        print(f"Accuracy: {self.total_online_tta_acc.compute()}".replace('.',','))
        if self.open_flag:
            h_score, known_acc, unknown_acc = self.total_online_tta_hscore.compute()
            print(f"H-Score: {h_score}".replace('.',','))
            print(f"Known Accuracy: {known_acc}".replace('.',','))
            print(f"Unknown Accuracy: {unknown_acc}".replace('.',','))
            self.log('H-Score', h_score)
            self.log('KnownAcc', known_acc)
            self.log('UnknownAcc', unknown_acc)

    def on_train_end(self):
        os.makedirs(os.path.join(self.trainer.log_dir, 'checkpoints'))
        torch.save({
            'backbone_state_dict': self.backbone.state_dict(),
            'feature_extractor_state_dict': self.feature_extractor.state_dict(),
            'classifier_state_dict': self.classifier.state_dict(),
        }, self.trainer.log_dir + '/checkpoints/adapted_ckpt.pt')

    def training_step(self, train_batch):
        opt = self.optimizers()
        self.on_test_model_eval()

        self.class_prototypes = self.class_prototypes.to(self.device)
        self.prototype_sum = self.prototype_sum.to(self.device)
        self.prototype_sample_counter = self.prototype_sample_counter.to(self.device)

        x, y, sample_ids = train_batch
        y = torch.where(y >= self.known_classes_num, self.known_classes_num, y)

        opt.zero_grad()

        # FORWARD
        y_hat, features = self.forward(x, apply_softmax=True)
        y_hat_aug, features_aug = self.forward(self.tta_transform(x))

        # ADAPTATION
        with torch.no_grad():
            pseudo_label_idx, pseudo_label = self.generate_pseudo_labels(y_hat, y, sample_ids)
            pseudo_label = pseudo_label.to(self.device)
            pseudo_label_idx = pseudo_label_idx.to(self.device)
        known_idx = torch.where(pseudo_label != self.known_classes_num)[0].to(self.device)
        unknown_idx = torch.where(pseudo_label == self.known_classes_num)[0].to(self.device)

        if not self.use_source_prototypes:
            with torch.no_grad():
                # Group the features by their pseudo-labels
                unique_labels = torch.unique(pseudo_label[known_idx])  # Get unique classes in the batch

                for label in unique_labels:
                    # Get indices of all samples with the current label
                    label_idx = (pseudo_label[known_idx] == label).nonzero(as_tuple=True)[0]

                    # Average the features of all samples for this class
                    class_features = features[pseudo_label_idx[known_idx[label_idx]]]  # Features for this label
                    avg_class_feature = torch.mean(class_features, dim=0)  # Averaging features of the same class

                    # Get the current class prototype
                    current_prototype = self.class_prototypes[label]

                    # Update using EMA with the averaged class feature
                    if torch.all(current_prototype == 0):
                        # Initialize with the first occurrence of this class
                        self.class_prototypes[label] = avg_class_feature
                    else:
                        alpha = 0.001
                        self.class_prototypes[label] = alpha * avg_class_feature + (1 - alpha) * current_prototype

        y_hat_entropy = -torch.matmul(y_hat, torch.log(y_hat.T)) / torch.log(torch.tensor(self.known_classes_num))
        y_hat_entropy = torch.diagonal(y_hat_entropy)

        if self.loss_type == 'contrastive+entropy':
            if len(known_idx) != 0:
                cl_known_features = torch.cat([torch.unsqueeze(self.class_prototypes[pseudo_label[known_idx]], dim=1),
                                        torch.unsqueeze(features[pseudo_label_idx[known_idx]], dim=1),
                                        torch.unsqueeze(features_aug[pseudo_label_idx[known_idx]], dim=1)],
                                        dim=1)
                unknown_idx = torch.where(pseudo_label == self.known_classes_num)[0]
                cl_unknown_features = torch.cat([torch.unsqueeze(features[pseudo_label_idx[unknown_idx]], dim=1),
                                                torch.unsqueeze(features_aug[pseudo_label_idx[unknown_idx]], dim=1)], dim=1)
                con_loss = self.contrastive_loss(cl_known_features, labels=pseudo_label[known_idx],
                                                confident_unknown_features=cl_unknown_features)

                entropy_loss = y_hat_entropy[pseudo_label_idx[known_idx]].mean() -\
                            y_hat_entropy[pseudo_label_idx[unknown_idx]].mean()
                loss = con_loss + self.lbd * entropy_loss
                self.manual_backward(loss)
                self.log('tta_loss', loss, on_epoch=True, prog_bar=True)
            else:
                loss = None

        elif self.loss_type == 'cross-entropy':
            if len(pseudo_label_idx) != 0:
                one_hot_pseudo_label = torch.zeros_like(y_hat[pseudo_label_idx])
                if len(known_idx) != 0:
                    one_hot_pseudo_label[known_idx, pseudo_label[known_idx]] = 1
                if len(unknown_idx) != 0:
                    one_hot_pseudo_label[unknown_idx] = 1 / self.known_classes_num
                loss = torch.sum(-one_hot_pseudo_label * torch.log(y_hat[pseudo_label_idx] + 1e-5), dim=-1).mean()
                self.manual_backward(loss)
                self.log('tta_loss', loss, on_epoch=True, prog_bar=True)
            else:
                loss = None
        opt.step()


        # PREDICTION
        with torch.no_grad():
            pred = torch.where(y_hat_entropy.detach() <= self.rejection_threshold, torch.argmax(y_hat.detach(), dim=1),
                               self.known_classes_num).to(self.device)
            self.total_online_tta_acc(pred, y)
            self.log('tta_acc', self.total_online_tta_acc, on_epoch=True, prog_bar=True)
            if self.open_flag:
                self.total_online_tta_hscore.update(pred, y)
