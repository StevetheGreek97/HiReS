from HiReS.pipeline import Pipeline
from HiReS.config import Settings
input_path = '/media/steve/UHH_EXT/Pictures/transfer_2961247_files_8d6ee684/'
model_path = '/home/steve/Desktop/NonofYaBuisness/zenodo/DaphnAI.pt'


cfg = Settings(
    conf=0.58, imgsz=1024, device="cpu",
    chunk_size=(1024, 1024), overlap=300,
    edge_threshold=0.01, iou_thresh=0.7
)

Pipeline(cfg).run(
    input_path=input_path,
    model_path=model_path, 
    output_dir="/home/steve/Desktop/tester/", 
    workers=5
)