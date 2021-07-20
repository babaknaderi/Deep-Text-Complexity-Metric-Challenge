# Deep Text Complexity Metric

Predicting the complexity of German sentences on a 7-step Likert Scale using a fine-tuned version of the GBERT model [gbert-base](https://huggingface.co/deepset/gbert-base). 

---

## Downloading the Model

Executing 

```
$ python3 get_model.py
```

will download the model and unpack its contents into the `model` folder.



## Expected Dataset Format

This implementation expects a certain dataset format. The dataset has to be a `.csv` file and needs to have the following two columns:
`sent_id` and `sentence`. If the dataset should be used for training, it also needs to have a `MOS` column with the target values.



## Using the Model

After downloading the model you can use it by executing 

```
$ python code/eval.py  --input dataset.csv
```

in your terminal, where `dataset.csv` is your dataset file. 
The results will be written to a `eval.csv` file.


## Training the Model

Execute

```
$ python code/train.py  --input dataset.csv --test_split 0.1
```

in your terminal, where `dataset.csv` is your dataset file 
and `test_split` is the ratio of items in the dataset used for evaluation.
The fine-tuned model will be saved to `model/deepset-gbert-base-finetuned`.

## License

MIT License

Copyright 2021 Max Reinhard & Faraz Maschhur

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
