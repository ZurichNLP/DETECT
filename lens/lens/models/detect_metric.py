import pandas as pd
import ast
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union
import wandb

from lens.lens.models.regression_metric_multi_ref import RegressionMetricMultiReference
from lens.lens.models.utils import Prediction, Target
from ..modules import FeedForward



class DetectMetric(RegressionMetricMultiReference):

    def __init__(self,
                alpha: float = 0.7,
                beta: float = 0.3,
                prediction_type: str = "single",  # "single", "multi"
                add_features: bool = False,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)
        
        self.alpha = alpha
        self.beta = beta
        self.prediction_type = prediction_type
        self.add_features = add_features
              
        #Parameters that are tested in train_lens but defined elsewhere:
        # batch_size, default = 4
        # pretrained_model= "xml-roberta"
        # dropout default = 0.1, used 0.3
        # hidden_sizes = [2304, 768]
        # activations = "Tanh" - not changed
        # final_activation = None - not to change so that it predicts z-scores!
        # encoder_learning_rate: float = 1e-05,
        # learning_rate: float = 3e-05,
        
        # Determine output dimensions
        self.output_dim = 3 if prediction_type == "multi" else 1
        
        # Setup model architecture based on configuration
        self._setup_model()

    def _setup_model(self):
        """Setup model architecture based on configuration parameters."""
        if self.add_features and self.prediction_type == "multi":
            # Custom architecture with features (german_lens_multi variant)
            self.embedding_proj = nn.Sequential(
                nn.Linear(self.encoder.model.config.hidden_size * 7, 128),
                getattr(nn, self.hparams.activation)(),
                nn.Dropout(self.hparams.dropout),
            )

            self.feature_proj = nn.Sequential(
                nn.Linear(7, 32),
                getattr(nn, self.hparams.activation)(),
                nn.Dropout(self.hparams.dropout),
            )

            self.estimator = nn.Sequential(
                nn.Linear(128 + 32, 64),
                nn.Linear(64, self.output_dim),
            )
            
        else:
            # Standard FeedForward architecture (original and multi_orig variants)
            self.estimator = FeedForward(
                in_dim=self.encoder.output_units * 7,
                hidden_sizes=self.hparams.hidden_sizes,
                activations=self.hparams.activations,
                dropout=self.hparams.dropout,
                final_activation=self.hparams.final_activation,
                out_dim=self.output_dim
            )

    def read_training_data(self, path: str):
        """Method that reads the training data (a csv file) and returns a list of
        samples.

        Returns:
            List[dict]: List with input samples in the form of a dict
        """
        df = pd.read_csv(path)
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["ref"] = df["ref"].apply(lambda x: ast.literal_eval(x))
        
        columns = ["src", "mt", "ref"]
        
        if self.add_features:
            columns.append("features")
            df["features"] = df["features"].apply(lambda x: np.array(ast.literal_eval(x), dtype="float16"))
            
        if self.prediction_type == "multi":
            df["scores"] = df["ind_scores"].apply(lambda x: np.array(ast.literal_eval(x), dtype="float16"))
        else:
            df["scores"] = df["avg_zscore"].astype("float16")
            
        columns.append("scores")
        df = df[columns]
        return df.to_dict("records")

    def read_validation_data(self, path: str):
        """Method that reads the validation data (a csv file) and returns a list of
        samples.

        Returns:
            List[dict]: List with input samples in the form of a dict
        """
        df = pd.read_csv(path)
        columns = ["src", "mt", "ref"]
        
        # If system in columns we will use this to calculate system-level accuracy
        if "system" in df.columns:
            columns.append("system")
            df["system"] = df["system"].astype(str)

        if self.add_features:
            columns.append("features")
            #df['features'] = df[['src', 'mt', 'ref']].apply(lambda x: calculate_features(x), axis = 1)
            df["features"] = df["features"].apply(lambda x: np.array(ast.literal_eval(x), dtype="float16"))

        if self.prediction_type == "multi":
            df["scores"] = df["ind_scores"].apply(lambda x: np.array(ast.literal_eval(x), dtype="float16"))
        else:
            df["scores"] = df["avg_zscore"].astype("float16")
            
        columns.append("scores")
        df = df[columns]
        df["src"] = df["src"].astype(str)
        df["mt"] = df["mt"].astype(str)
        df["ref"] = df["ref"].apply(lambda x: ast.literal_eval(x))
        return df.to_dict("records")

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], stage: str = "train"
    ) -> Union[
        Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]
    ]:
        inference = (stage == "predict")
        sample = {k: [dic[k] for dic in sample] for k in sample[0]}
        all_inputs = []

        # Determine if we need to handle features
        has_features = self.add_features and 'features' in sample

        if has_features:
            zip_items = zip(sample['src'], sample['mt'], sample['ref'], sample['features'])
        else:
            zip_items = zip(sample['src'], sample['mt'], sample['ref'])

        for items in zip_items:
            if has_features:
                src, mt, refs, features = items
            else:
                src, mt, refs = items
                features = None

            if isinstance(refs, str):
                refs = [refs]
            num_refs = len(refs)

            src_batch = [src] * num_refs
            mt_batch = [mt] * num_refs

            src_inputs = self.encoder.prepare_sample(src_batch)
            mt_inputs = self.encoder.prepare_sample(mt_batch)
            ref_inputs = self.encoder.prepare_sample(refs)

            src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
            mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
            ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}

            input_dict = {
                **src_inputs,
                **mt_inputs,
                **ref_inputs,
            }

            if has_features:
                input_dict["features"] = torch.tensor(np.array([features] * num_refs), dtype=torch.float)

            all_inputs.append(input_dict)

        if inference:
            return all_inputs

        targets = {
            "scores": torch.tensor(np.array(sample["scores"]), dtype=torch.float)
        }
        return all_inputs, targets

    def forward(self, **kwargs) -> Prediction:
        """Forward pass that handles different model architectures."""
        src_emb = self.get_sentence_embedding(kwargs['src_input_ids'], kwargs['src_attention_mask'])
        mt_emb = self.get_sentence_embedding(kwargs['mt_input_ids'], kwargs['mt_attention_mask'])
        ref_emb = self.get_sentence_embedding(kwargs['ref_input_ids'], kwargs['ref_attention_mask'])

        if self.add_features and self.prediction_type == "multi":
            # Custom architecture with features
            diff_ref = torch.abs(mt_emb - ref_emb)
            diff_src = torch.abs(mt_emb - src_emb)
            prod_ref = mt_emb * ref_emb
            prod_src = mt_emb * src_emb

            combined_emb = torch.cat((src_emb, mt_emb, ref_emb, prod_ref, diff_ref, prod_src, diff_src), dim=1)
            return self.estimate_with_features(combined_emb, kwargs['features'])
        else:
            # Standard architecture
            return self.estimate(src_emb, mt_emb, ref_emb)

    def estimate(self, src_sentemb: torch.Tensor, mt_sentemb: torch.Tensor, ref_sentemb: torch.Tensor) -> Prediction:
        """Standard estimation method using FeedForward."""
        diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        diff_src = torch.abs(mt_sentemb - src_sentemb)
        prod_ref = mt_sentemb * ref_sentemb
        prod_src = mt_sentemb * src_sentemb

        embedded_sequences = torch.cat(
            (src_sentemb, mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src),
            dim=1,
        )
        return Prediction(scores=self.estimator(embedded_sequences))

    def estimate_with_features(self, embedded_sequences: torch.Tensor, features: torch.Tensor) -> Prediction:
        """Estimation method for custom architecture with features."""
        emb_proj = self.embedding_proj(embedded_sequences)
        feat_proj = self.feature_proj(features)
        combined = torch.cat([emb_proj, feat_proj], dim=1)
        raw_output = self.estimator(combined)
        #scaled_output = self.output_rescale(raw_output)
        return Prediction(scores=raw_output)

    def compute_loss(self, prediction: Prediction, target: Target) -> torch.Tensor:
        """Compute loss based on prediction type."""
        x = prediction['scores']
        y = target['scores']
        eps = 1e-8

        if self.prediction_type == "single":
            # Single score loss (original variant)
            x_norm = (x - x.mean()) / (x.std() + eps)
            y_norm = (y - y.mean()) / (y.std() + eps)
            corr = torch.mean(x_norm * y_norm)
            pearson_loss = 1 - corr
            mse_loss = torch.mean((x - y) ** 2)
        else:
            # Multi-score loss
            x_norm = (x - x.mean(dim=0)) / (x.std(dim=0) + eps)
            y_norm = (y - y.mean(dim=0)) / (y.std(dim=0) + eps)
            corr = torch.mean(x_norm * y_norm, dim=0)
            pearson_loss = 1 - corr.mean()
            mse_loss = torch.mean((x - y) ** 2)

        return self.alpha * mse_loss + self.beta * pearson_loss

    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
        batch_nb: int,
    ) -> torch.Tensor:
        """Training step that handles different prediction types with reference-adaptive loss."""
        batch_input, batch_target = batch
        
        batch_predictions = []
        batch_targets = []
        
        for one_sample, target in zip(batch_input, batch_target['scores']):
            prediction = self.forward(**one_sample)
            ref_count = one_sample['ref_input_ids'].size(0)
            K = min(self.topk, ref_count)
            
            if self.prediction_type == "single":
                # Single score prediction - select top-k scores
                topk_scores = torch.topk(prediction['scores'].view(-1), K).values
                topk_targets = target.repeat(K)
                
                batch_predictions.append(topk_scores)
                batch_targets.append(topk_targets)
                
            else:  # multi prediction
                # Multi-dimensional prediction - select top-k based on mean across dimensions
                if prediction.scores.dim() == 1:
                    # Handle case where only one reference exists
                    prediction_scores = prediction.scores.unsqueeze(0)
                else:
                    prediction_scores = prediction.scores
                    
                # Select top-k references based on mean across the three dimensions
                aggregate_scores = prediction_scores.mean(dim=1)
                topk_indices = torch.topk(aggregate_scores, K).indices
                
                # Get the multi-dimensional scores for selected references
                selected_predictions = prediction_scores[topk_indices]  # shape: [K, num_dimensions]
                
                # Repeat target for each selected reference
                repeated_targets = target.unsqueeze(0).repeat(K, 1)  # shape: [K, num_dimensions]
                
                batch_predictions.append(selected_predictions)
                batch_targets.append(repeated_targets)
        
        # Combine all predictions and targets for batch-level loss computation
        if self.prediction_type == "single":
            # For single prediction: concatenate into 1D tensors
            combined_predictions = torch.hstack(batch_predictions)
            combined_targets = torch.hstack(batch_targets)
            
            batch_prediction = {'scores': combined_predictions}
            batch_target_dict = {'scores': combined_targets}
            
        else:  # multi prediction
            # For multi prediction: concatenate along batch dimension
            combined_predictions = torch.cat(batch_predictions, dim=0)  # shape: [total_K, num_dimensions]
            combined_targets = torch.cat(batch_targets, dim=0)  # shape: [total_K, num_dimensions]
            
            batch_prediction = {'scores': combined_predictions}
            batch_target_dict = {'scores': combined_targets}
        
        # Compute loss using the combined batch
        loss_value = self.compute_loss(batch_prediction, batch_target_dict)
        
        # Handle encoder unfreezing (applies to both single and multi modes)
        if (
            self.nr_frozen_epochs < 1.0
            and self.nr_frozen_epochs > 0.0
            and batch_nb > self.epoch_total_steps * self.nr_frozen_epochs
        ):
            self.unfreeze_encoder()
            self._frozen = False
        
        self.log("train_loss", loss_value, on_step=True, on_epoch=True)
        return loss_value

    def _safe_scalar_metrics(self, metrics_dict):
        """Helper method to safely convert metrics to scalars."""
        safe_metrics = {}
        for k, v in metrics_dict.items():
            if isinstance(v, (float, int)):
                safe_metrics[k] = v
            elif isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    safe_metrics[k] = v.item()
                else:
                    safe_metrics[f"{k}_mean"] = v.mean().item()
            elif isinstance(v, np.ndarray):
                if v.ndim == 0 or v.size == 1:
                    safe_metrics[k] = float(v)
                elif v.ndim == 1 or v.ndim == 2:
                    safe_metrics[f"{k}_mean"] = float(np.mean(v))
        return safe_metrics

    def validation_step(self, batch, batch_nb: int, dataloader_idx: int) -> None:
        """Handle validation epoch end for different prediction types."""
        if self.prediction_type == "single":
            # Use parent class logic for single prediction
            return super().validation_step(batch, batch_nb, dataloader_idx)
    
        else:
            batch_input, batch_target = batch
            all_preds = []
            all_targets = []

            for one_sample, target in zip(batch_input, batch_target['scores']):
                prediction = self.forward(**one_sample)
                pred_mean = prediction.scores.mean(dim=0)
                all_preds.append(pred_mean)
                all_targets.append(target)

            preds_tensor = torch.stack(all_preds)
            targets_tensor = torch.stack(all_targets)

            loss_value = self.compute_loss({'scores': preds_tensor}, {'scores': targets_tensor})
            self.log("val_loss", loss_value, on_step=True, on_epoch=True)

            # Update metric objects
            if dataloader_idx == 0:
                self.train_metrics.update(preds_tensor, targets_tensor)
            elif dataloader_idx == 1:
                self.val_metrics.update(preds_tensor, targets_tensor)

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        """Handle validation epoch end for different prediction types."""
        if self.prediction_type == "single":
            # Use parent class logic for single prediction
            return super().on_validation_epoch_end(*args, **kwargs)
        
        else:  # multi prediction
            train_metrics = self._safe_scalar_metrics(self.train_metrics.compute())
            self.log_dict(train_metrics, prog_bar=False)
            self.train_metrics.reset()

            val_results = self._safe_scalar_metrics(self.val_metrics.compute())
            self.val_metrics.reset()
            self.log_dict(val_results, prog_bar=True)

            log_data = {**train_metrics, **val_results}
            wandb.log({f"val_epoch/{k}": v for k, v in log_data.items()})