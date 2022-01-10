import argparse

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained',default='True', help='bool value to indicate weather to use pretrained weight for encoder')
    parser.add_argument('--dataset_path', help = 'path to dataset')
    parser.add_argument('--dims', default= [512,512,512,512,512,512,512,512,128], help = 'list indicating number of hidden units for each layer of projection head')
    parser.add_argument('--num_class', default = 3)
    parser.add_argument('--encoder', default = 'resnet18')
    parser.add_argument('--learninig_rate', default=0.03)
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight_decay', default=0.00003)
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--num_gpus', default=1)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--input_size', default=256)
    parser.add_argument('--log_dir', default=r'tb_logs') 
    parser.add_argument('--log_dir_name', default=r'exp1')
    parser.add_argument('--checkpoint_filename', default=r'weights') 
    parser.add_argument('--monitor_checkpoint', default=r'train_loss') 
    parser.add_argument('--monitor_checkpoint_mode', default=r'min') 

  
    args = parser.parse_args()
    return args