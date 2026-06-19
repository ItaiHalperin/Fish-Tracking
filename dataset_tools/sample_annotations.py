import argparse
import shutil
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Sample frames for classification annotation")
    parser.add_argument("--crops_dir", type=str, default="crops", help="Directory containing the extracted crops")
    parser.add_argument("--output_dir", type=str, default="annotation_samples", help="Output directory for samples")
    parser.add_argument("--samples", type=int, default=50, help="Number of 5-frame sequences to sample")
    parser.add_argument("--gap", type=int, default=15, help="Gap in frames between the images (default 15)")
    args = parser.parse_args()
    
    crops_path = Path(args.crops_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning for valid 5-frame sequences (gap={args.gap}) in {crops_path}")
    valid_sequences = []
    
    if not crops_path.exists():
        print(f"Crops directory {crops_path} does not exist.")
        return
        
    for video_dir in crops_path.iterdir():
        if not video_dir.is_dir():
            continue
            
        for id_dir in video_dir.iterdir():
            if not id_dir.is_dir() or not id_dir.name.startswith("id_"):
                continue
                
            # Read all frames
            frames = []
            for img_file in id_dir.glob("frame_*.jpg"):
                try:
                    # extract number from 'frame_123.jpg'
                    frame_num = int(img_file.stem.split('_')[1])
                    frames.append(frame_num)
                except ValueError:
                    pass
                    
            frames_set = set(frames)
            
            # Find valid sequences
            for n in frames:
                # We need n, n-gap, n-2*gap, n-3*gap, n-4*gap
                req_frames = [n - i * args.gap for i in range(4, -1, -1)]
                
                if all(f in frames_set for f in req_frames):
                    valid_sequences.append({
                        "video": video_dir.name,
                        "track_id": id_dir.name,
                        "frames": req_frames,
                        "folder": id_dir
                    })
                    
    print(f"Found {len(valid_sequences)} valid sequences across all track IDs.")
    
    if len(valid_sequences) == 0:
        print("Not enough continuous frames to sample. Try decreasing the gap or ensuring long enough tracks.")
        return
        
    # Sample randomly
    num_to_sample = min(args.samples, len(valid_sequences))
    sampled = random.sample(valid_sequences, num_to_sample)
    
    for i, seq in enumerate(sampled):
        sample_id = f"sample_{i+1:04d}"
        sample_dir = output_path / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the files and retain original names
        for f_num in seq["frames"]:
            src = seq["folder"] / f"frame_{f_num}.jpg"
            dst = sample_dir / f"frame_{f_num}.jpg"
            shutil.copy2(src, dst)
            
        # Save a metadata file
        meta_file = sample_dir / "meta.txt"
        with open(meta_file, 'w') as f:
            f.write(f"Video: {seq['video']}\n")
            f.write(f"Track ID: {seq['track_id']}\n")
            f.write(f"Frames: {seq['frames']}\n")
            
    print(f"Successfully sampled {num_to_sample} sequences into {output_path}")

if __name__ == '__main__':
    main()
