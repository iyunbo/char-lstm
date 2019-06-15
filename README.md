# char-lstm
build an character level LSTM on GPU for generating some fun text

When I was learning RNN from
https://github.com/udacity/deep-learning-v2-pytorch/blob/master/recurrent-neural-networks/char-rnn, I was so interested 
in the subject of intelligent generator of text. So I decided to build my own engineered version of char-rnn. I reused 
some code from the original char-rnn, but I mostly improve them by adding unit tests, structuring with modules and tuning
the model in order to make this program useful for more use cases.

### Run the program
```bash
python run.py [-h] [-f FILE] -p PRIME -n COUNT [-s SEQ_LENGTH]
```

- -f FILE: the text file train model from. (optional)
- -p PRIME: the prefix of generated text
- -n COUNT: the number of characters of generated text
- -s SEQ_LENGTH: the length of a meaningful sequence of characters, ex: the number of characters in a phrase. (optional)

examples:
```bash
python run.py -p "Levin said" -n 1600
```
```bash
python run.py -f mytext.txt -p "Levin said" -n 1600 -s 100
```

This program will load pre-trained model if it exists, otherwise, the training process may take **10~30 minutes**.

### License
Licensed by [MIT License](LICENSE)