nbpe=15000
bpemode=unigram
mkdir -p ${bpemode}
dict=${bpemode}/${bpemode}${nbpe}_units.txt
bpemodel=${bpemode}/${bpemode}${nbpe}
echo -e "<unk> 1\n<en> 2\n<ar> 3\n<es> 4\n<fr> 5\n<zh> 6" > "${dict}" # <unk> must be 1, 0 will be used for "blank" in CTC
python spm_train.py --split_digits=True --input=all_labels.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
python spm_encode.py --model=${bpemodel}.model --output_format=piece < all_labels.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+6}' >> ${dict}