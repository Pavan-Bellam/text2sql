import os
import boto3
from transformers import TrainerCallback

class S3UploadCallback(TrainerCallback):
    def __init__(self, s3_bucket, s3_prefix):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client('s3')
        
    def on_save(self, args, state, control, **kwargs):
        """Event called after a checkpoint save."""
        # The last checkpoint path is usually in state.best_model_checkpoint 
        # or we can infer it from the output_dir
        
        # Find the latest checkpoint directory
        # Checkpoints are saved as "checkpoint-X"
        ckpt_dir = f"checkpoint-{state.global_step}"
        local_path = os.path.join(args.output_dir, ckpt_dir)
        
        if os.path.isdir(local_path):
            print(f"\n[S3 Callback] Uploading {ckpt_dir} to s3://{self.s3_bucket}/{self.s3_prefix}/{ckpt_dir}...")
            self._upload_dir(local_path, f"{self.s3_prefix}/{ckpt_dir}")
            print(f"[S3 Callback] Upload finished.")

    def _upload_dir(self, local_path, s3_path):
        """Recursively uploads a directory to S3."""
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                # Calculate relative path to keep folder structure
                relative_path = os.path.relpath(local_file, local_path)
                s3_file = os.path.join(s3_path, relative_path)
                
                self.s3_client.upload_file(local_file, self.s3_bucket, s3_file)