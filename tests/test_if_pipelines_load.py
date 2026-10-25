import pickle

def test_pipeline_loading():

    pipelines = [
        "models/PIPELINE_PD_v012_LR_cal_none.pkl",
        "models/PIPELINE_PD_v012_LR_cal_shift-test.pkl",
        "models/PIPELINE_PD_v012_GB_cal_shift-test.pkl",
        "models/PIPELINE_LGD_v012_amount.pkl",
    ]

    for pipeline_path in pipelines:

        print('testing pipeline:', pipeline_path)

        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)

        assert pipeline is not None, f"Failed to load {pipeline_path}"