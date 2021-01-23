## Access data and construct a datablock with fastai

First of all, we need to set the path to our dataset and read it in with pandas. Here I'm using a dataframe as an example:

```
df = pd.read_csv('filepath')
df.head()
```

<img src="C:\Users\richa\AppData\Roaming\Typora\typora-user-images\image-20210123140220632.png" alt="image-20210123140220632" style="zoom: 50%;" />

To access the `fname` only, we could use `DataFrame.iloc` or `df['fname']`

```
df.iloc[:,0]
df['fname']
```

<img src="C:\Users\richa\AppData\Roaming\Typora\typora-user-images\image-20210123140818797.png" alt="image-20210123140818797" style="zoom:50%;" />

Now for classification problems, we always want our data labelled and have it divided into training and validation set. The `Datablock` object of Fastai can do this automatically for us.

The `Datablock` is a generic container to quickly build `Datasets` and `DataLoaders`:

```
dblock = Datablock()
dblock?
```

<img src="C:\Users\richa\AppData\Roaming\Typora\typora-user-images\image-20210123143154190.png" alt="image-20210123143154190" style="zoom:50%;" />

We could then build a datasets object with a source. With ` .train` and `.valid`, it randomly splits the whole dataset into training set with 80% of data and validation set with 20% of data.

```
dsets = dblock.datasets(df)
len(dsets.train),len(dsets.valid) #number of training and validation sets
x, y = dsets[0]
x,y
```

<img src="C:\Users\richa\AppData\Roaming\Typora\typora-user-images\image-20210123143845530.png" alt="image-20210123143845530" style="zoom:50%;" />

What if we want the file name to be something else and the labels to be split up?

```
def get_x(r): return r['fname']
def get_y(r): return r['labels'].split('')
dblock = Datablock(blocks = (ImageBlock,MultiCategoryBlcok)
							,get_x = get_x,get_y = get_y) #To have it opened as image with 																multilabel
dsets = dblock.datasets(df)
dsets.train[0]
```

<img src="C:\Users\richa\AppData\Roaming\Typora\typora-user-images\image-20210123162422439.png" alt="image-20210123162422439" style="zoom:50%;" />

Now we can put our `datasets`into `dataloaders`. What `dataloadesrs`does is basically collating the items from `datasets`into a mini batch. While we are doing so, we need to also make sure that the images are of the same size.

```
dblock = Datablock.(blocks(ImageBlock,MultiCategoryBlcok),
				  get_x = get_x,get_y = get_y,
				  item_tfms = RandomResizedCrop(128,min_scale = 0.35))
dls = dblock.Dataloaders(df)
```

Display a few:

```
dls.show_batch(nrows=3, ncols=3)
```

<img src="C:\Users\richa\AppData\Roaming\Typora\typora-user-images\image-20210123163851911.png" alt="image-20210123163851911" style="zoom:67%;" />