cd ../scripts/

# for shuffled
python main.py --data_name=recog_results_14-Oct_SHUFFLED_run3
echo "shuffled done"

# for not shuffled
python main.py --data_name=recog_results_14-Oct_notshuffled-run1
python main.py --data_name=recog_results_14-Oct_notshuffled-run3
echo "not shuffled done"

# for random segmentation
python main.py --data_name=recog_results_randomseg-run3_dict
python main.py --data_name=recog_results_randomseg-run1_dict
echo "random segmentation done"
