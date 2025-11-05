import os
from ultralytics import YOLO

class YOLOSegPredictor:
    def __init__(self, model_path, output_dir="output"):
        """
        Initialize the YOLOSegPredictor class.

        :param model_path: Path to the YOLO segmentation model.
        :param output_dir: Directory to save the outputs.
        """
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        

    def predict(self, 
                image_dir: str, 
                conf: float = 0.5,
                imgsz: int = 1024, 
                device = 'cpu' ) -> None:
        """
        Run YOLO segmentation inference on a folder of images.

        Args:
            image_dir (str): Directory containing images to predict on.
            conf (float): Confidence threshold for filtering predictions.
        """

        for result in self.model(image_dir, 
                                 conf=conf, 
                                 verbose=False, 
                                 stream=True, 
                                 visualize = False, 
                                 imgsz=imgsz, 
                                 device=device):
            image_name = os.path.splitext(os.path.basename(result.path))[0]
            result.save_txt(f'{self.output_dir}/{image_name}.txt', save_conf= True)
            #result.save(f'{self.img_ann_dir}/{image_name}.jpeg', line_width=1)
            #result.save_crop(f'{self.ann_dir}/croped')
            #result.visualize(f'{self.ann_dir}/feature')
    

def main():
    pass            

if __name__ == "__main__":
    
    main()
