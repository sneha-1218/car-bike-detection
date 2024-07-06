

# car-bike-detection

## Aim and Objectives
## Aim
The primary aim of this project is to develop a robust object detection model to identify cars and bikes in images and videos using YOLOv5. 
## objectives 
- Cloning the YOLOv5 repository and setting up the environment.
- Training the YOLOv5 model on a custom dataset.
- Running the detection model on test images and videos.
- Demonstrating the model's performance and potential applications.

## Abstract
This project focuses on using the YOLOv5 model to detect cars and bikes in various visual media. By leveraging the YOLOv5 architecture, known for its speed and accuracy, the project aims to provide a reliable solution for real-time object detection. The trained model is capable of identifying cars and bikes in images, videos, and live feeds.

## Introduction
Object detection is a crucial aspect of computer vision, enabling systems to recognize and locate objects within an image 
or video. YOLOv5 (You Only Look Once) is a state-of-the-art object detection model that balances accuracy and speed, making it suitable for real-time applications. This project utilizes YOLOv5 to detect cars and bikes, providing a practical solution for various applications such as traffic monitoring and autonomous driving.
## Literature Review
Several studies and projects have demonstrated the effectiveness of YOLO models in object detection. The YOLOv5 model, in particular, has shown significant improvements in terms of speed and accuracy over its predecessors. This project builds on the existing research and applies YOLOv5 to a specific use case: detecting cars and bikes in images and videos.

## Methodology
### Installation
1. **Clone the YOLOv5 Repository**
    ```sh
    !git clone https://github.com/ultralytics/yolov5
    %cd yolov5
    %pip install -qr requirements.txt
    %pip install -q roboflow
    ```
2. **Set Up Environment**
    ```python
    import torch
    import os
    from IPython.display import Image, clear_output

    print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
    os.environ["DATASET_DIRECTORY"] = "/content/datasets"
    ```
3. **Install Roboflow and Download Dataset**
    ```python
    !pip install roboflow
    from roboflow import Roboflow
    rf = Roboflow(api_key="XWjxucFvUZIKB7eeA2x6")
    project = rf.workspace("sneha-w2mic").project("car_bike-tvhus")
    version = project.version(2)
    dataset = version.download("yolov5")
    ```

### Running
1. **Train the Model**
    ```sh
    !python train.py --img 416 --batch 4 --epochs 100 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
    ```
2. **Run Detection**
    - On Test Images:
        ```sh
        !python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.1 --source {dataset.location}/test/images
        ```
    - On a Specific Image URL:
        ```sh
        !python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.1 --source https://stat.overdrive.in/wp-content/odgallery/2022/08/63812_2022_Honda_CB300F_DLX_PRO_1_468x263.jpg
        ```
    - On Additional Test Images:
        ```sh
        !python detect.py --weights /content/yolov5/yolov5s.pt --img 416 --conf 0.1 --source {dataset.location}/test/images



## Advantages
- **Real-Time Detection**: Capable of detecting objects in real-time with high accuracy.
- **Versatility**: Can be applied to various types of media including images, videos, and live feeds.
- **Ease of Use**: Simple setup and execution process with detailed instructions provided.

## Applications
- Traffic Monitoring: Identifying and counting vehicles in real-time.
- Autonomous Driving: Enhancing vehicle awareness by detecting surrounding vehicles.
- Security Systems: Monitoring parking lots and identifying unauthorized vehicles.

## Future Scope
- **Improvement of Detection Accuracy**: Further training with larger and more diverse datasets.
- **Integration with Other Systems**: Combining with other sensors and systems for enhanced functionality.
**Real-World Testing**: Deploying the model in real-world scenarios to test its robustness and reliability.

## Conclusion
This project successfully demonstrates the use of YOLOv5 for detecting cars and bikes in various visual media. The trained model provides a reliable solution for real-time object detection with potential applications in traffic monitoring, autonomous driving, and security systems.

## References
- YOLOv5: https://github.com/ultralytics/yolov5
- Roboflow: https://roboflow.com
- Various articles and research papers on object detection and YOLO models.

### Demo 




https://github.com/sneha-1218/car-bike-detection/assets/174138540/8595567c-8d14-46ef-ae10-14d314aa0c11

Link:-  https://youtu.be/rX6MXfttWPs


