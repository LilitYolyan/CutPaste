import argparse

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--pretrained', , default='True', help='bool value to indicate weather to use pretrained weight for encoder')
    parser.add_argument('--dataset_path', default=r'/media/annamanasyan/Data/Manufacturing/MVTec') 
    parser.add_argument('--dims', default= [512,512,512,512,512,512,512,512,128], help = 'list indicating number of hidden units for each layer of projection head')
    parser.add_argument('--num_classes', default = 3)
    parser.add_argument('--learninig_rate', default=0.03)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', default=0.001)
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--num_gpus', default=1)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--input_size', default=448)
    parser.add_argument('--save_path', default=r'./') 
  
    args = parser.parse_args()
    return args