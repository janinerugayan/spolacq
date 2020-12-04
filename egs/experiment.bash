cd ../scripts/

# for shuffled
# echo "START shuffled"
# python main.py --data_name=recog_results_14-Oct_SHUFFLED_run3
# echo "shuffled done"

# # for not shuffled
# echo "START not shuffled"
# python main.py --data_name=recog_results_14-Oct_notshuffled-run1
# python main.py --data_name=recog_results_14-Oct_notshuffled-run3
# echo "not shuffled done"
#
# # for random segmentation
# echo "START random segmentation"
# python main.py --data_name=recog_results_randomseg-run3_dict
# python main.py --data_name=recog_results_randomseg-run1_dict
# echo "random segmentation done"

# rerun this batch:
echo "========= START re-run of not-shuffled run2 ========="
python main.py --data_name=recog_results_14-Oct_notshuffled-run2
echo "========= START re-run of shuffled run2 ========="
python main.py --data_name=recog_results_14-Oct_SHUFFLED_run2
echo "========= START re-run of randomseg run2 ========="
python main.py --data_name=recog_results_randomseg-run2_dict

echo "========= START re-run of 03-Dec dictionary ========="
python main.py --data_name=recog_results_03-Dec_notshuffled
python main.py --data_name=recog_results_03-Dec_SHUFFLED
echo "========= re-run done ========="
