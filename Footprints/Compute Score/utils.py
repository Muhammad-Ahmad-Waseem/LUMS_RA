import shapely.wkt
import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm
import os
import sys
# from . import iou
from fiona.errors import DriverError
from fiona._err import CPLE_OpenFailedError

def calculate_iou(pred_poly, test_data_GDF):
    """Get the best intersection over union for a predicted polygon.

    """
    # Fix bowties and self-intersections
    if not pred_poly.is_valid:
        pred_poly = pred_poly.buffer(0.0)

    precise_matches = test_data_GDF[test_data_GDF.intersects(pred_poly)]

    iou_row_list = []
    for _, row in precise_matches.iterrows():
        # Load ground truth polygon and check exact iou
        test_poly = row.geometry
        # Ignore invalid polygons for now
        if pred_poly.is_valid and test_poly.is_valid:
            intersection = pred_poly.intersection(test_poly).area
            union = pred_poly.union(test_poly).area
            # Calculate iou
            iou_score = intersection / float(union)
        else:
            iou_score = 0
        row['iou_score'] = iou_score
        iou_row_list.append(row)

    iou_GDF = gpd.GeoDataFrame(iou_row_list)
    return iou_GDF

class Evaluator():
    """Object to test IoU for predictions and ground truth polygons.

    Attributes
    ----------
    ground_truth_fname : str
        The filename for the ground truth CSV or JSON.
    ground_truth_GDF : :class:`geopandas.GeoDataFrame`
        A :class:`geopandas.GeoDataFrame` containing the ground truth vector
        labels.
    ground_truth_GDF_Edit : :class:`geopandas.GeoDataFrame`
        A copy of ``ground_truth_GDF`` which will be manipulated during
        processing.
    proposal_GDF : :class:`geopandas.GeoDataFrame`
        The proposal :class:`geopandas.GeoDataFrame`, added using
        ``load_proposal()``.

    Arguments
    ---------
    ground_truth_vector_file : str
        Path to .geojson file for ground truth.

    """

    def __init__(self, ground_truth_vector_file, proposal_vector_file):
        # Load Ground Truth : Ground Truth should be in csv format
        self.load_files(ground_truth_vector_file, proposal_vector_file)
        self.ground_truth_fname = ground_truth_vector_file
        self.ground_truth_sindex = self.ground_truth_GDF.sindex  # get sindex
        # create deep copy of ground truth file for calculations
        self.ground_truth_GDF_Edit = self.ground_truth_GDF.copy(deep=True)
        self.proposal_GDF_copy = self.proposal_GDF.copy(deep=True)

    def load_files(self, ground_truth_vector_file, proposal_vector_file, poly_col_key='PolygonWKT_Pix'):
        """Load in the ground truth geometry data and proposal csv geometry data.
        Does not return anything, just sets some attributes of the class
        for ground truth geopandas dataframe
        Loads the ground truth vector data into the ``Evaluator`` instance.

        """
        truth_data = pd.read_csv(ground_truth_vector_file)
        print(truth_data.columns)
        self.ground_truth_GDF = gpd.GeoDataFrame(
            truth_data, geometry=[
                shapely.wkt.loads(truth_row[poly_col_key])
                for idx, truth_row in truth_data.iterrows()])
        self.ground_truth_GDF['Detected'] = 0
        # force calculation of spatialindex
        self.ground_truth_sindex = self.ground_truth_GDF.sindex
        # create deep copy of ground truth file for calculations
        self.ground_truth_GDF_Edit = self.ground_truth_GDF.copy(deep=True)
        pred_data = pd.read_csv(proposal_vector_file)
        self.proposal_GDF = gpd.GeoDataFrame(
            pred_data, geometry=[
                shapely.wkt.loads(pred_row[poly_col_key])
                for idx, pred_row in pred_data.iterrows()
            ]
        )
        # set arbitrary (meaningless) values otherwise
        self.proposal_GDF['Detected'] = 0


    def eval(self, type='iou'):
        pass

    def __repr__(self):
        return 'Evaluator {}'.format(os.path.split(
            self.ground_truth_fname)[-1])

    def eval_iou_spacenet_csv(self, miniou=0.5, min_area=0):
        """Evaluate IoU between the ground truth and proposals in CSVs

        """
        # Get List of all ImageID in both ground truth and proposals
        scoring_dict_list = []
        id_cols = 2
        #print("Here")
        print("Starting...")
        print("Total polygons in GT: {}".format(len(self.ground_truth_GDF_Edit)))
        self.ground_truth_GDF_Edit = self.ground_truth_GDF_Edit[
            self.ground_truth_GDF_Edit.area >= min_area
            ]
        print("Total polygons in GT after area filter: {}".format(len(self.ground_truth_GDF_Edit)))

        print("Total polygons in Prediction: {}".format(len(self.proposal_GDF_copy)))
        self.proposal_GDF_copy = self.proposal_GDF_copy[self.proposal_GDF_copy.area
                                              >= min_area]
        print("Total polygons in Prediction after area filter: {}".format(len(self.proposal_GDF_copy)))

        print(self.ground_truth_GDF_Edit.head())
        print(self.proposal_GDF_copy.head())
        # tqdm(dataloader, desc="CelebA_Data", file=sys.stdout) as iterator
        #  tqdm(self.proposal_GDF_copy.iterrows()):
        with tqdm(self.proposal_GDF_copy.iterrows(), desc="Evaluating Area".format(miniou), file=sys.stdout) as iterator:
            for _, pred_row in iterator:
                if pred_row.geometry.area > 0:
                    #print("Here")
                    pred_poly = pred_row.geometry
                    iou_GDF = calculate_iou(pred_poly, self.ground_truth_GDF_Edit)
                    # Get max iou
                    if not iou_GDF.empty:
                        max_index = iou_GDF['iou_score'].idxmax(axis=0,
                                                                skipna=True)
                        max_iou_row = iou_GDF.loc[max_index]
                        if max_iou_row['iou_score'] > miniou:
                            self.proposal_GDF.loc[pred_row.name, 'Detected'] \
                                = 1
                            self.ground_truth_GDF.loc[max_iou_row.name, 'Detected'] \
                                = 1
                            self.ground_truth_GDF_Edit \
                                = self.ground_truth_GDF_Edit.drop(
                                    max_iou_row.name, axis=0)
                s = "IOU threshold: {}".format(miniou)
                iterator.set_postfix_str(s, refresh=False)

        if self.proposal_GDF_copy.empty:
            TruePos = 0
            FalsePos = 0
        else:
            if 'Detected' in self.proposal_GDF.columns:
                TruePos = self.proposal_GDF[
                    self.proposal_GDF['Detected'] == 1].shape[0]
                FalsePos = self.proposal_GDF[
                    self.proposal_GDF.area > min_area][
                    self.proposal_GDF['Detected'] == 0].shape[0]

            else:
                print("Detected field {} missing in predictions")
                TruePos = 0
                FalsePos = 0

        if self.ground_truth_GDF.empty:
            FalseNeg = 0
        else:
            if 'Detected' in self.ground_truth_GDF.columns:
                FalseNeg = self.ground_truth_GDF[
                    self.ground_truth_GDF.area > min_area][
                    self.ground_truth_GDF['Detected'] == 0].shape[0]
            else:
                print("Detected field {} missing in GT")
                FalseNeg = 0

        print("Total TP: {}".format(TruePos))
        print("Total FP: {}".format(FalsePos))
        print("Total FN: {}".format(FalseNeg))

        if float(TruePos+FalsePos) > 0:
            Precision = TruePos / float(TruePos + FalsePos)
        else:
            Precision = 0
        if float(TruePos + FalseNeg) > 0:
            Recall = TruePos / float(TruePos + FalseNeg)
        else:
            Recall = 0
        if Recall * Precision > 0:
            F1Score = 2*Precision*Recall/(Precision+Recall)
        else:
            F1Score = 0

        score_calc = {'iou_threshold': miniou,
                      'TruePos': TruePos,
                      'FalsePos': FalsePos,
                      'FalseNeg': FalseNeg,
                      'Precision': Precision,
                      'Recall':  Recall,
                      'F1Score': F1Score
                      }
        scoring_dict_list.append(score_calc)

        return scoring_dict_list, self.ground_truth_GDF, self.proposal_GDF

