"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from text_recognition.modules.transformation import TPS_SpatialTransformerNetwork
from text_recognition.modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from text_recognition.modules.sequence_modeling import BidirectionalLSTM
from text_recognition.modules.prediction import Attention
from text_recognition import text_recognition_config

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.stages = {'Trans': text_recognition_config.TRANSFORMATION, 'Feat': text_recognition_config.FEATUREEXTRACTION,
                       'Seq': text_recognition_config.SEQUENCEMODELING, 'Pred': text_recognition_config.PREDICTION}

        """ Transformation """
        if text_recognition_config.TRANSFORMATION == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=text_recognition_config.NUM_FIDUCIAL, I_size=(text_recognition_config.IMAGE_HEIGHT, text_recognition_config.IMAGE_WIDTH), I_r_size=(text_recognition_config.IMAGE_HEIGHT, text_recognition_config.IMAGE_WIDTH), I_channel_num=text_recognition_config.INPUT_CHANNEL)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if text_recognition_config.FEATUREEXTRACTION == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(text_recognition_config.INPUT_CHANNEL, text_recognition_config.OUTPUT_CHANNEL)
        elif text_recognition_config.FEATUREEXTRACTION == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(text_recognition_config.INPUT_CHANNEL, text_recognition_config.OUTPUT_CHANNEL)
        elif text_recognition_config.FEATUREEXTRACTION == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(text_recognition_config.INPUT_CHANNEL, text_recognition_config.OUTPUT_CHANNEL)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = text_recognition_config.OUTPUT_CHANNEL  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if text_recognition_config.SEQUENCEMODELING == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, text_recognition_config.HIDDEN_SIZE, text_recognition_config.HIDDEN_SIZE),
                BidirectionalLSTM(text_recognition_config.HIDDEN_SIZE, text_recognition_config.HIDDEN_SIZE, text_recognition_config.HIDDEN_SIZE))
            self.SequenceModeling_output = text_recognition_config.HIDDEN_SIZE
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if text_recognition_config.PREDICTION == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, text_recognition_config.num_class)
        elif text_recognition_config.PREDICTION == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, text_recognition_config.HIDDEN_SIZE, text_recognition_config.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=25)

        return prediction
