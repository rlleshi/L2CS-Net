import pathlib
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from face_detection import RetinaFace

from .utils import prep_input_numpy, getArch
from .results import GazeResultContainer


class Pipeline:

    def __init__(
        self, 
        weights: pathlib.Path, 
        arch: str,
        device: str = 'cpu', 
        include_detector:bool = True,
        confidence_threshold:float = 0.5
        ):

        # Save input parameters
        self.weights = weights
        self.include_detector = include_detector
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Create L2CS model
        self.model = getArch(arch, 90)
        self.model.load_state_dict(torch.load(self.weights, map_location=device))
        self.model.to(self.device)
        self.model.eval()

        # Create RetinaFace if requested
        if self.include_detector:

            if device.type == 'cpu':
                self.detector = RetinaFace()
            else:
                self.detector = RetinaFace(gpu_id=device.index)

        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = [idx for idx in range(90)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.device)

    def step(self, frame: np.ndarray) -> GazeResultContainer:

        # Creating containers
        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []

        if self.include_detector:
            faces = self.detector(frame)

            if faces is not None:
                for box, landmark, score in faces:

                    # Apply threshold
                    if score < self.confidence_threshold:
                        continue

                    # Extract safe min and max of x,y
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)

                    # Save data
                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)

                # Predict gaze
                pitch, yaw = self.predict_gaze(np.stack(face_imgs))

            else:

                pitch = np.empty((0,1))
                yaw = np.empty((0,1))

        else:
            pitch, yaw = self.predict_gaze(frame)

        if self.include_detector:
            results = GazeResultContainer(
                pitch=pitch,
                yaw=yaw,
                bboxes=np.stack(bboxes),
                landmarks=np.stack(landmarks),
                scores=np.stack(scores)
            )
        else:
            results = GazeResultContainer(
                pitch=pitch,
                yaw=yaw,
                bboxes=None,
                landmarks=None,
                scores=None
            )

        return results

    def predict_gaze(
            self,
            frame: Union[np.ndarray, torch.Tensor],
            *,
            detach: bool = False,
            to_numpy: bool = False
        ):
        if isinstance(frame, np.ndarray):
            img = prep_input_numpy(frame, self.device)   # 1,C,H,W float32
        elif isinstance(frame, torch.Tensor):
            img = frame.to(self.device)
        else:
            raise TypeError("frame must be np.ndarray or torch.Tensor")

        gaze_yaw, gaze_pitch = self.model(img)
        yaw_prob   = self.softmax(gaze_yaw)
        pitch_prob = self.softmax(gaze_pitch)

        # convert to degrees
        idx = self.idx_tensor
        yaw  = (yaw_prob   * idx).sum(dim=1) * 4 - 180
        pitch = (pitch_prob * idx).sum(dim=1) * 4 - 180

        if detach:
            yaw   = yaw.detach()
            pitch = pitch.detach()

        if to_numpy:
            yaw   = (yaw.cpu()   * np.pi/180.0).numpy()
            pitch = (pitch.cpu() * np.pi/180.0).numpy()
        else:
            yaw   = yaw * np.pi/180.0
            pitch = pitch * np.pi/180.0

        return pitch, yaw