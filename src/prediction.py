import torch
from src.utils import *
from src.data_load.data_loader import *
from src.data_load.KnowledgeGraph import KnowledgeGraph
from torch.utils.data import Dataset, DataLoader
from torch import tensor, from_numpy, no_grad, save, load, arange
from torch.autograd import Variable
from src.model.original_model import TransE as base_model
from src.model.model_test import TransE as model_test
import argparse

class Predict:
    def __init__(self, args, kg):
        '''set device'''
        self.args = args
        torch.cuda.set_device(int(args.gpu))
        _ = torch.tensor([1]).cuda()
        self.device = _.device  # Set self.device directly
        self.args.device = self.device  # Ensure args.device is set

        self.kg = kg  # Assuming 'kg' is an instance of KnowledgeGraph
        self.batch_size = args.val_batch_size
        self.args.snapshot = 4
        self.args.snapshot_test =4
        '''prepare data'''
        self.dataset = TestDataset(args, kg)
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      collate_fn=self.dataset.collate_fn,
                                      generator=torch.Generator().manual_seed(int(args.seed)),
                                      pin_memory=True)
        
        if self.args.lifelong_name == 'base_model':
            self.model = base_model(self.args, self.kg)
        elif self.args.lifelong_name == 'model_test':
            self.model = model_test(self.args, self.kg)
        else:
            raise ValueError("Invalid model name specified.")
        # self.model.switch_snapshot(prediction = True)
        checkpoint = torch.load(args.input_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])

    def forward(self):
        self.model.eval()
        num = 4
        results = dict()
        sr2o = self.kg.snapshots[self.args.snapshot_num].sr2o_all

        '''start evaluation'''
        for step, batch in enumerate(self.data_loader):
            sub, rel, obj, label = batch
            sub = sub.to(self.device)
            rel = rel.to(self.device)
            obj = obj.to(self.device)
            label = label.to(self.device)
            num += len(sub)

            stage = 'Valid' if self.args.valid else 'Test'

            '''link prediction'''
            pred = self.model.predict(sub, rel, stage=stage)  # [100,10513]
            b_range = torch.arange(pred.size()[0], device=self.device)
            target_pred = pred[b_range, obj]  # [100]
            pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
            pred[b_range, obj] = target_pred

            '''rank all candidate entities'''
            ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                b_range, obj]
            '''get results'''
            ranks = ranks.float()
            results['count'] = torch.numel(ranks) + results.get('count', 0.0)
            results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
            results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
            for k in range(10):
                results['hits{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                    'hits{}'.format(k + 1), 0.0)

        count = float(results['count'])
        for key, val in results.items():
            results[key] = round(val / count, 4)
        return results

def main():
    parser = argparse.ArgumentParser(description="Run prediction")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file')
    parser.add_argument('--lifelong_name', type=str, default='LKGE', help='Competitor name or LKGE')
    parser.add_argument('--snapshot_num', type=int, default=5, help='The snapshot number of the dataset')
    parser.add_argument('--dataset', type=str, default='HYBRID', help='dataset name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--loss_name', type=str, default='Margin', help='Margin: pairwise margin loss')
    parser.add_argument('--train_new', action='store_false', help='Train on new facts; defaults to training on all seen facts if not set')
    parser.add_argument('--skip_previous', action='store_false', help='Skip previous training steps and snapshot_only models if set')
    parser.add_argument('--epoch_num', type=int, default=200, help='Maximum epoch number')
    parser.add_argument('--margin', type=float, default=8.0)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--val_batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--emb_dim', type=int, default=200)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--neg_ratio', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=55, help='random seed, 11 22 33 44 55 for our experiments')
    parser.add_argument('--data_path', type=str, default='/fs/scratch/PCS0273/lijing/LKGE/data/HYBRID/')
    parser.add_argument('--log_path', type=str, default='/fs/scratch/PCS0273/lijing/LKGE/logs/')
    parser.add_argument('--num_layer', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    # Set the device attribute on args before passing it to KnowledgeGraph
    torch.cuda.set_device(int(args.gpu))
    _ = torch.tensor([1]).cuda()
    args.device = _.device

    # Create a KnowledgeGraph instance (kg) here according to your specific setup
    kg = KnowledgeGraph(args)
    
    predictor = Predict(args, kg)
    results = predictor.forward()
    print(results)

if __name__ == "__main__":
    main()
