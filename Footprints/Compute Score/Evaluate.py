"""Script for executing eval for SpaceNet challenges."""
import argparse
import pandas as pd
from utils import Evaluator
#supported_challenges = ['off-nadir', 'spacenet-buildings2']
# , 'spaceNet-buildings1', 'spacenet-roads1', 'buildings', 'roads']

parser = argparse.ArgumentParser(
    description='Evaluate SpaceNet Competition CSVs')
parser.add_argument('--proposal_csv', '-p', type=str,
                    help='Proposal CSV')
parser.add_argument('--truth_csv', '-t', type=str,
                    help='Truth CSV')
# parser.add_argument('--challenge', '-c', type=str,
#                     default='off-nadir',
#                     choices=supported_challenges,
#                     help='SpaceNet Challenge eval type')
parser.add_argument('--output_file', '-o', type=str,
                    default='Evaluation',
                    help='Output file To write results to CSV')
parser.add_argument('--iou_threshold', '-i', type=float,
                    help='Thresholding for iou score')


if __name__ == '__main__':
    args = parser.parse_args()
    truth_file = args.truth_csv
    prop_file = args.proposal_csv

    evaluator = Evaluator(ground_truth_vector_file=args.truth_csv,
                          proposal_vector_file = args.proposal_csv)
    results, gt_gdf, pr_gdf = evaluator.eval_iou_spacenet_csv(miniou=args.iou_threshold,
                                              min_area=20)
    results_DF_Full = pd.DataFrame(results)
    gt_evaluated = pd.DataFrame(gt_gdf)
    pr_evaluated = pd.DataFrame(pr_gdf)

    if args.output_file:
        print("Writing summary results to {}".format(
            args.output_file.rstrip('.csv') + '.csv'))
        results_DF_Full.to_csv(args.output_file.rstrip('.csv') + '_results.csv',
                          index=False)
        gt_evaluated.to_csv(args.output_file.rstrip('.csv') + '_gt.csv',
                          index=False)
        pr_evaluated.to_csv(args.output_file.rstrip('.csv') + '_pr.csv',
                            index=False)