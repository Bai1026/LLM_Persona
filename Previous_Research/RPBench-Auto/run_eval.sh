# # run character eval 3 times, write in for loop
# for i in {1..5}
# do
#     python run_character_eval.py -m1 gan_model
# done

# echo "Character eval done"

python run_character_eval.py -m1 gan_model
python run_character_eval.py -m1 gan_model
python run_character_eval.py -m1 gan_model
# python run_character_eval.py -m1 gan_model
# python run_character_eval.py -m1 gan_model
# python run_character_eval.py -m1 gan_model