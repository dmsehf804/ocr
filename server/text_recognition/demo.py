import string
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from text_recognition.utils import CTCLabelConverter, AttnLabelConverter
from text_recognition.dataset import RawDataset, AlignCollate
from text_recognition.model import Model
from text_recognition import text_recognition_config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" model configuration """
if 'CTC' in text_recognition_config.PREDICTION:
    converter = CTCLabelConverter(text_recognition_config.LABLE)
else:
    converter = AttnLabelConverter(text_recognition_config.LABLE)
text_recognition_config.num_class = len(converter.character)

model = Model()
model = torch.nn.DataParallel(model).to(device)

# load model
print('loading pretrained model from %s' % text_recognition_config.SAVE_MODEL)
model.load_state_dict(torch.load(text_recognition_config.SAVE_MODEL, map_location=device))


def text_recognition(image):

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=text_recognition_config.IMAGE_HEIGHT, imgW=text_recognition_config.IMAGE_WIDTH)
    demo_data = RawDataset(image)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=192,
        shuffle=False,
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([25] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, 25 + 1).fill_(0).to(device)

            if 'CTC' in text_recognition_config.PREDICTION:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                t = time.time()
                preds = model(image, text_for_pred, is_train=False)
                print('text recog infer: ', time.time() - t)
                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            log = open(f'./log_demo_result.txt', 'a')
            dashed_line = '-' * 80
            head = f'{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in text_recognition_config.PREDICTION:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                print(type(img_name), type(pred), type(confidence_score))
                print(f'{pred:25s}\t{confidence_score:0.4f}')
                log.write(f'{pred:25s}\t{confidence_score:0.4f}\n')

            log.close()
            return str(pred)