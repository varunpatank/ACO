import os
import sys
from scipy.io import savemat

os.chdir(os.path.dirname(__file__))
sys.path.append('..')

from SMBJClassifier import DPP, model, PLA

def main():
    filename = 'COVID_Strand_Source.txt'
    Expfile = 'COVID_Exp_parameter.mat'
    data, Amp, Freq, RR, Vbias = DPP.readInfo(filename, Expfile)

    # Make sure PLA.py currently uses: abs(slope) < cs
    # PLA.createCondTrace_PLA(data, Amp, Freq, RR, Vbias)

    runs = [
        {
            "name": "Alpha_lessThanAco",
            "labels": ["Alpha_MM1", "Alpha_MM2", "Alpha_PM"],
            "groups": [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]]
        },
        {
            "name": "Beta_lessThanAco",
            "labels": ["Beta_MM1", "Beta_MM2", "Beta_MM3", "Beta_PM"],
            "groups": [[16,17,18], [19,20,21], [22,23,24], [25,26,27]]
        },
        {
            "name": "Delta_lessThanAco",
            "labels": ["Delta_MM1", "Delta_MM2", "Delta_PM"],
            "groups": [[51,52,53,54], [55,56,57,58], [59,60,61,62]]
        }
    ]

    approach = 6
    sampleNum = 30
    os.makedirs('./Result', exist_ok=True)

    for run in runs:
        print("Running:", run["name"])
        conf_mat = model.runClassifier(
            approach,
            data,
            RR,
            run["groups"],
            len(run["labels"]),
            sampleNum
        )

        savemat(
            f'./Result/{run["name"]}_A6_H30.mat',
            {
                'conf_mat': conf_mat,
                'group_Label': run["labels"],
                'approach': approach,
                'sampleNum': sampleNum,
                'filter_type': 'abs_slope_less_than_Aco'
            }
        )

if __name__ == '__main__':
    main()
