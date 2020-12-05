cd ../scripts/

echo "========= START RL for 05-Dec dictionary ========="
python main.py --data_name=recog_results_05-Dec_SHUFFLED_run1
python main.py --data_name=recog_results_05-Dec_SHUFFLED_run2
python main.py --data_name=recog_results_05-Dec_SHUFFLED_run3
echo "========= RL - SHUFFLED - done ========="

echo "========= START RL for 05-Dec dictionary ========="
python main.py --data_name=recog_results_05-Dec_notshuffled_run1
python main.py --data_name=recog_results_05-Dec_notshuffled_run2
python main.py --data_name=recog_results_05-Dec_notshuffled_run3
echo "========= RL - not shuffled - done ========="
