# 17

Hi everybody and welcome to Lesson 17  of Practical Deep Learning for Coders. 
I'm really excited about what we're going  to look at over the next lesson or two. 
It's actually been turning out really  well, much better than I could have hoped.  So I can't wait to dive in. Before I do, I'm just going  
to mention a couple of minor changes that  I made to our miniai library this week. 
One was I went back to our Callback class  in the learner notebook and I did decide in  
the end to add a dunder getattr to it that just adds these four attributes. 
And for these four attributes,  it passes it down to self.learn.  So in a callback, you'll be able to  refer to model to get self.learn.model, 
opt will be self.learn.opt, batch will be  self.learn.batch, epoch will be self.learn.epoch. 
You can change these, you know,   you could subclass the callback and  add your own to underscore forward, 
or you could remove things from  underscore forward or whatever.  But I felt like these four things I access  a lot and I was sick of typing self.learn. 
And then I added one more property, which is  in a callback there'll be a self.training, 
which saves from typing self.learn.model.training. Since we have model, you can get rid of the learn. 
But still, I mean, you so often  have to check the training.  Now you can just get self.training in a callback. So that was one change I made. 
The second change I made was I found myself  getting a bit bored of adding TrainCB every time. 
So what I did was I took the four training  methods from the Momentum learner subclass, 
and I've moved them into a TrainLearner  subclass along with zero_grad. 
So now MomentumLearner actually inherits  from TrainLearner and just adds momentum. 
There's kind of a quirky momentum method and  changes zero_grad to do the momentum thing. 
So yeah, so we'll be using TrainLearner  quite a bit over the next lesson or two. 
So TrainLearner is just a Learner  which has the usual training. 
It's exactly the same that fastai2 has or  you'd have in most PyTorch training loops. 
And obviously by using this, you lose the  ability to change these with a callback.  So it's a little bit less flexible. Okay, so those are little changes. 
And then I made some changes to what we looked  at last week, which is the activations notebook. 
And specifically, Okay, so I added a HooksCallback. 
So previously, we had a Hooks class, and it  didn't really require too much ceremony to use, 
but I thought we could make  it even simpler and a bit more   fastai-ish or miniai-ish by  putting hooks into a callback. 
So this callback, as usual, you pass a function  that's going to be called for your hook. 
And you can optionally pass it a filter  as to what modules you want to hook. 
And then in before_fit, it will  filter the modules in the Learner. 
And this is one of these things we can now get rid  of, we don't need the .learn here because model is   one of the four things we have a shortcut to. And then here, we're going to create the Hooks  
object and put it in hooks.  And so one thing that's convenient here is the  hook function. Now you don't have to worry ‒and  
we can get rid of learn.model‒, you don't have  to worry about checking in your hook functions,   whether in training or not, it  always checks whether in training. 
And if so, it calls that hook function you passed  in. And after it finishes, it removes the hooks. 
And you can iterate through the hooks and get the  length of the hooks because it just passes these   iterators and length down to self.hooks. So to show you how this works,  
we can create a HooksCallback. We can use the same append_stats. 
And then we can run the model. 
And so as it's training, what we're going to be  able to do is, yeah, we can now then, here we go. 
So we just added that as an extra  callback to our fit function.  I don't remember if we had the extra callbacks  before. I'm not sure we did. So just to explain. 
I just added extra callbacks here   in the fit function and we're just  adding any extra callbacks here. 
So then now we've got that callback that we  created because we can iterate through it and so  
forth, we can just iterate through that callback  as if it's hooks and plot in the usual way. 
So that's a convenient little thing.  I think it's convenient thing I added. 
OK. And then I took our colorful dimension  stuff, which is defined when I came up with  
a few years ago and decided to wrap  all that up in a callback as well.  So I've actually subclassed here our hooks  callback to create an ActivationStats. 
And what that's going to do is it's going to  use this append_stats, which appends the means,  
the standard deviations and the histograms. And oh, and I changed that very slightly. 
Also, the thing which creates these kind  of dead plots, I changed it to just get the  
ratio of the very first, very smallest  histogram bin to the rest of the bins. 
So these are really kind of more  like very dead at this point.  So these graphs look a little bit different. 
OK, so, yeah, so I subclassed  the hooks callback and and  
yeah, added the colorful dimension method,  dead chart method and a plot stats method. 
So to see them at work, if we want to  get the activations on all of the convs. 
Then we train our model and then we can just call.  And so we've added, created our ActivationStats. We've added that as an extra callback and then. 
And then, yeah, then we can  call color_dim to get that plot,   dead_chart to get that plot and  plot_stats to get that shot plot. 
So now we have absolutely no excuse for  not getting all of these really fantastic,  
informative visualizations of what's going  on inside our model, because it's literally   as easy as adding one line of code. And just putting that in your callbacks. 
So I really think that couldn't be easier. And so I hope you're, even for models you thought   you know, we're training really well. Why don't you try using this? 
Because you might be surprised  to discover that they're not.  OK, so those are some changes. Pretty minor, but hopefully useful. 
Trying to get 90% accuracy on Fashion-MNIST
And so today, and over the next lesson  or two, we're going to look at trying to  
get to a important milestone, which is  to try to get Fashion-MNIST training to  
accuracy of 90 percent or more, which  is certainly not the end of the road. 
But it's not bad if we look at paperswithcode.  There's so 90 percent would be a 10 percent error. So there's folks that have got down to 3  
or 4 percent error in the very  best, which is very impressive.  But, you know, 10 percent  error wouldn't be way off. 
What's in this paper leaderboard? I don't know how far we'll get eventually, but,  
without using even any architectural  changes, no ResNets or anything.  We're trying to get into the 10 percent error. All right, so, let's so the first few cells are  
just copied from earlier. And so. 
Here's a ridiculously simple model. I like all I did here was I said, OK,   well, the very first convolution is  taking a 9 by 9 by 1 channel input. 
So we should have compressed  it at least a little bit.  So I made it 8 channels  output for the convolution. 
And then I just doubled it to 16,  doubled it to 32, doubled it to 64. 
And so that's going to get to  a, that will be as you say,   14 by 14 image, 7 by 7 , a 4 by 4, a 2 by 2. And then this one gets us to a 1 by 1. 
So, of course, we get the 10 digits. So there was no thought at all behind,   really, this architecture. This pure, just pure convolutional architecture. 
And remember, this Flatten at the end is necessary  to get rid of the unit axis that we end up with  
because this is a 1 by 1. OK, so let's do a learning   rate finder on this very simple model. And what I found was that this model is. 
And, you know, this situation is so bad that  when I tried to use the learning rate finder   kind of in the usual way, which would be just to say,  
you know, start at 1e-5 or  1e-4, say, and then run it. 
It kind of looks ridiculous. It's  impossible to see what's going on. 
So if you remember, we added that  multiplier, we called it lr_mult or  
gamma is what they called it in PyTorch. So we ended up calling it gamma.  So I dialed that way down to make  it much more gradual, which means I  
have to dial up the starting learning rate. And only then did I manage even to get the  
learning rate finder to tell us anything useful. OK, so. So there we are. 
So that's our learning rate finder. Come back to these three later. 
So I tried using a learning rate of 0.2, and  after trying a few different values, 0.4, 0.1,  
0.2 seems about the highest we can get up to. Even this actually is too high, I found.  
Much lower and it didn't train much at all. You can see what happens if I do.   It starts training and then it kind of. Yeah, we lose it, which is unfortunate. And you  
can see that in the colorful dimension plot. We get this classic. You know,  
getting activations, crashing,  getting activations, crashing.  And you can kind of say the key problem  here really is that we don't have 0 mean. 
Standard deviation 1 layers at the start. So we certainly don't keep them throughout.  
And this is a problem.  That is something I got to  mention, by the way, is. 
When you're training stuff in Jupyter notebooks,  this is just a new thing we've just added. 
Jupyter notebooks and GPU memory
If you get… you can easily run out of memory,  GPU memory, and there's two reasons it turns out  
why you can particularly run out of GPU memory  if you run a few cells in a Jupyter notebook. 
The first is that kind of for your convenience. Jupyter notebook. —you might may or may not know  
this— actually stores the results  of your previous few evaluations. 
If you just type underscore, it tells  you the very last thing you evaluated. 
And you can do more underscores  to go backwards further in time.  Or you can also use. Oh, you can also use numbers  to get the out 16, for example, would be _16. 
Now, the reason this is an issue is that  if one of your outputs is a big CUDA tensor  
and you've shown it in a cell, that's going  to keep that GPU memory basically forever. 
And so that's a bit of a problem. So if you are running out of memory,  
one thing you'd want to do is clean out  all of those underscore blah things. 
I found that there's actually some function that  nearly does that in the IPython source code. 
So I copied the important bits  out of it and put it in here.  So if you call clean_ipython_hist, it will  —don't worry about the lines of code at all. 
This is just a thing that you can  use to get back that GPU memory. 
The second thing, which Piotr figured out in  the last week or so, is that you also have…  
if you have a CUDA error at any point or  even any kind of exception at any point, 
then the exception object is actually stored  by Python and any tensors that were allocated  
anywhere in that trace, in that traceback,  will stay allocated basically forever. 
And again, that's a big problem. So I created this clean trace back function  
based on Piotr’s code, which gets rid of that. So this is particularly problematic because if  
you have a CUDA out of memory error and then  you try to rerun it, you'll still have a CUDA   out of memory error because all the memory that  was allocated before is now in that trace back. 
So basically, any time you get a CUDA out of  memory error or any kind of error with memory,  
you can call clean_mem and that will  clean the memory in your trace back.  It will clean the memory  used in your Jupyter history. 
Do a garbage collect. Empty the CUDA cache and that will basically,  
should give you a totally clean GPU. You don't have to restart your notebook. 
Autoencoder or Classifier
OK, so Sam asked a very good question in the chat. So just to remind you guys, yes, we did start. 
He's asking, I thought we were  training an autoencoder or are   we training a classifier or what? So we started doing this autoencoder  
back in notebook 8 and we decided, oh, this is  we don't have the tools to make this work yet. 
So let's go back and create the  tools and then come back to it.  So in creating the tools,  we're doing a classifier. 
We try to make a really good  Fashion-MNIST classifier.  Well, we try to create tools  which hopefully have a side effect 
giving us a really good classifier and then using  those tools, we hope that will allow us to create  
a really good autoencoder. So, yes, we're kind of like  
gradually unwinding and we'll come back to  where we were actually trying to get to. 
So that's why we're doing this this classifier. The techniques and library pieces we're building  
will be all very necessary. OK, so why do we need a 0 mean,  
Why do we need a mean of 0 and standard deviation of 1?
1 standard deviation? Why do we need that?.   And B, how do we get it? So first of all, on the why. 
So if you think about what a neural net  does, a deep learning net specifically. 
It takes an input and it puts it  through a whole bunch of matrix   multiplications and of course there are  activation functions sandwiched in there. 
Don't worry about the activation functions. That doesn't change the argument.  So let's just imagine we start with some bunch of,  
some matrix. Right?. 
Imagine the 50 deep neural net. So 50 deep neural net, basically,  
if we ignore the activation functions is taking  the previous input and doing a matrix multiply by  
some, initially, some random weights. So these are all.  Yeah, these are just a bunch of random weights. And these are actually randn. 
And is mean 0, variance 1. And if we run this. 
After 50 times of multiplying by a  matrix by matrix by matrix by matrix. 
We end up with. NaNs.  That's no good. So that might be that our matrix,  
the numbers in our matrix, were too big. So each time we multiply the numbers   were getting bigger and bigger and bigger. So maybe we should make them a bit smaller. Okay,  
so let's try using, in the matrix we are  playing by, let's try multiplying by 0.01. 
And we multiply that lots of  times. Oh, now we've got zeros.  Now, of course, mathematically speaking,  this isn't actually NaN. It's actually some  
really big number. Mathematically speaking, this  isn't really zero. It's a really small number.  But computers can't handle really, really  small numbers are really, really big numbers. 
So really, really big numbers eventually just  get called NaN and really, really small numbers   eventually just get called zero. So basically they get washed out.  
And in fact, even if you don't get a NaN  or even if you don't quite get a zero. 
The numbers that are extremely big.  The internal representation  has no ability to discriminate  
between even slightly similar numbers. Basically, in the way a floating point  
is stored, the further you get away from  zero, the less accurate the numbers are. 
So, yeah, this is a problem. So we have  to scale weight matrices exactly right. 
And we have to scale them in such  a way that the standard deviation   at every point stays at one  and the mean stays at zero. 
So there's actually a paper that describes how to  do this for multiplying lots of matrices together. 
And this paper basically just went  through. It's actually pretty simple math. 
Actually, let's see. What did they do? All right. 
Yeah, so they looked at gradients  and the propagation of gradients,   and they came up with a particular weight  initialization of using a uniform with. 
With one over root n as  the bounds of that uniform. 
And they studied basically what happened  with various different activation functions. 
And as a result, we now have this  way of initializing neural networks,  
which is called either Glorot  initialization or Xavier initialization. 
And, yeah, this is the amount that we scale  our initialization, our random numbers by,  
where nin is the number of inputs. So in our case, we have  
100 inputs. And so root 100 is 10. So 1/10 is 0.1. And so if we actually run that, if we start with  
our random numbers, and then we multiply  by random numbers times 0.1, which is,  
this is the Glorot initialization, you  can see we do end up with numbers that are  
actually reasonable. So that's pretty cool. 
So just, I mean, just some background in case  you're not familiar with some of these details. 
What exactly do we mean by variance?
What exactly do we mean by variance? So if we take a tensor, let's call it t  
and just put 1, 2, 4, 18 in it. The mean of that is simply  
the sum divided by the count. So that's 6.25. Now, we want to know,  
basically, we want to come up with a measure of  how far away each data point is from the mean. 
That tells you how much variation there is. If all  the data points are very similar to each other. 
So if you've got kind of like  a whole bunch of data points.  And they're all pretty similar to each other. Right?. Then the mean would be about here. Right?. 
And the average distance away of each  point from the mean is not very far. 
Whereas if you had dots which were  very widely spread all over the place. 
Right?. Then you might end up with the  same mean. But the distance from each point  
to the mean is now quite a long way. So that's what we want. We want some  
measure of kind of how far away the  points are on average from the mean. 
So here we could do that. We can take  our tensor, we can subtract the mean.  And then take the mean of that. Ah, that doesn't work. 
Because we've got some numbers  that are bigger than the mean and   some that are smaller than the mean. And so if you average them all out,  
then by definition you actually get zero.  So instead you could either square those  differences and that will give you something. 
And you could also take the square root of that  if you wanted to to get it back to the same  
kind of area. Or you could take the absolute differences. 
OK. So actually I'm doing this in  two steps here. So for the first one,   here it is on a different scale. And then add square root,  
get it on the same scale. So 6.87 and 5.88 are quite similar, right?  But they're mathematically not quite the  same. But they're both similar ideas. 
So this is the mean absolute difference.  And this is called the standard deviation.  And this is called the variance. 
So the reason that the standard deviation  is bigger than the mean absolute  
difference is because in our original data, one of the numbers is much bigger than the others. 
And so when we square it, that number  ends up having an outsized influence. 
And so that's a bit of an issue in general with  standard deviation and variance is that outliers  
like this have an outsized influence. So you've got to be a bit careful. 
OK. So here's the formula for the standard  deviation that's normally written as sigma. 
So it's just going to be each of our  data points minus the mean squared   plus the next data point minus  the mean squared, so forth, 
all the data points and then divide that by  the number of data points in square root. 
And OK. So one thing I point out here  is that the mean absolute deviation   isn't used as much as the standard deviation, because mathematicians find it difficult to use.  
But we're not mathematicians. We have computers so we can use it.  
OK, now, variance, we can  calculate like this, as we said,   the mean of the square of the differences. And if you feel like doing some math,  
you discover that actually this is  exactly the same, as you can see.  And this is actually nice because this is  showing that the mean of the squared data points  
minus the square of the mean of the  data points is also the variance. 
And this is very helpful because it means  you actually never have to calculate this. 
You can just calculate the mean. So  with just the data points on their own,   you can actually calculate the variance. This is a really nice shortcut. This  
is how we normally calculate variance. And so there is the LaTeX version, which,  
of course, I didn't write myself. I stole  from the Wikipedia LaTeX because I'm lazy. 
Now, there's a very, very similar idea.  
Covariance
Which is covariance and has already come  up a little bit in the first lesson or two. 
And particularly the extra math  lesson that was same and Tanishq did. 
And it's yes, a covariance tells you how much two  things vary, not just on their own, but together. 
And there's a definition here in math,   but I like code, so we'll see the  code. So here's our tensor again. 
Now we're going to want to have two things. So  let's create something called u, which is just   two times our tensor with a bit of randomness. So here it is. Now you can see that u and t  
are very closely correlated here. But they're not perfectly correlated.  
So the covariance tells us how  they vary together and separately. 
So we can take the, you can see this  exactly the same thing we had before. 
Each data point minus its mean. But  now we've got two different tensors. 
So we're also going to do the other one, the  other the other data points minus their mean.   And we multiply them together. So it's actually the same thing as standard  
deviation. But instead of deviation, it's kind  of like the covariance with itself in a sense. 
Right. And so that's a product we can calculate. 
And then what we then do is  we take the mean of that. 
And that gives us the covariance  between those two tensors. 
And you can see that's quite a high  number. And if we compare it to two  
things that aren't very related at all,  so it's quite a totally random tensor, v. 
So this is not related to t. And  we do exactly the same thing. 
So take the difference of t to its  means and v to its means and take the   mean of that. That's a very small number. And so you can see covariance is basically  
telling us how related are these two tensors. So covariance and variance are basically the  
same thing. But you kind of can think of you can  think of variance as being covariance with itself. 
And you can change this mathematical version,  which is the one we just created in code,  
to this version, just like we have for variance.  There's an easier to calculate version. 
Which, as you can see, gives  exactly the same answer. 
OK, so if you haven't done stuff with covariance  much before, you should experiment a bit with it  
by creating a few different plots  and experimenting with those. 
And finally, the Pearson correlation  coefficient, which is normally called  
r or rho, is just the covariance divided  by the product of the standard deviations. 
So you've seen, probably  seen that number many times.  There's just a scaled version of the same thing. 
Xavier Glorot initialization
OK, so with that in mind, here is how  Xavier init or Glorot init is derived. 
So when you do a matrix multiplication,  
for each of the yi's, we're adding  together all of these products. 
So we've got ai,0 times x0  plus ai,1 times x1, etc. 
And we can write that in sigma notation. So  we're adding up together all of the aik's  
with all of the xk's. This is the stuff that   we did in our first lesson of Part 2. And so here it is in pure Python code. 
And here it is in NumPy code. Now at the very beginning, our vector  
has a mean of about 0 and a standard deviation  of about 1 because that's what we asked for. 
That's what we asked for. That's a standard deviation of 1, mean of 0. 
That's what randn is. OK, so let's create some random  
numbers and we can confirm, yeah, they have a mean  of about 0 and a standard deviation of about 1. 
So if we chose weights for a,   that have a mean of 0, we can compute  the standard deviation quite easily. 
So let's do that. So 100 times,   let's try creating our x and let's try  creating something to multiply it by. 
And we'll do the matrix multiplication. And we're going to get the  
mean and… mean of the squares. 
And so that is very close to our matrix. 
So I won't go into, I mean, you can look at  it if you like, but basically as long as the   elements in a and x are independent, which  obviously they are because they're random, 
then we're going to end up with a mean of 0 and  a standard deviation of 1 for these products. 
And so we can try it if we create a random number,  
normally distributed random number and  then a second random number, multiply them   together and then do it a bunch of times. And you can see here we've got our 0, 1. 
So that's the reason why we need  this math dot square root 100. 
We don't normally worry about the  mathematical reasons why things are exactly.  But yeah, I thought I would just dive into this  one because sometimes it's fun to go through it. 
And so you can check out the paper  if you want to look at that in more   detail or experiment with these,  with these little simulations. 
Now the problem is that that doesn't work. It doesn't work for us because we use  
rectified linear units, which is not  something that Xavier Glorot looked at. 
Let's take a look. Let's create a  couple of matrices. This is 200 by 100. 
This is just a vector matrix  and a vector. This is 200. 
And then let's create a couple of weight matrices,  two weight matrices and two bias vectors. 
OK, so we've got some input data x's and y's and  we've got some weight matrices and bias vectors. 
So let's create a linear layer function,  which we've done lots of times before.  And let's start going through a little  neural net. You know, I'm mentioning this  
is the forward pass of our neural net. So we're going to apply our linear layer   to the x’s with our first set of weights  and our first set of biases and see what  
the mean and standard deviation is. OK, it's. About 0 and about 1. 
So that's good news. And the  reason why, is because we have  
100 inputs and we divided it by square root 100, just like Glorot told us to. And our second one  
has 50 inputs and we divide by square root of 50. And so this all ought to work right. And so  
far it is. But now we're going to  mess everything up by doing ReLU.  So ReLU, after we do a ReLU, look, we don't  have a 0 mean or 1 standard deviation anymore. 
So if we go through that and create it like a  deep neural network with Glorot initialization. 
But with a ReLU. Oh, dear. It's disappeared. It's all gone to 0. And you  
can see why, right? After a matrix multiply and  a ReLU, our means and variances are going down. 
And of course, they're going  down because a ReLU squishes it. 
ReLU and Kaiming He initialization
So I'm not going to worry about the math of why,   but a very important paper indeed  called “Delving Deep into Rectifiers: 
Surpassing Human-Level Performance  on ImageNet Classification”   by Kaiming He et al, came up with a new init, which is just like Glorot initialization. But  
you multiply the… remember the Glorot  initialization was one over root n. 
This one is root 2 over n. And again, n  is the number of inputs. So let's try it. 
So we've got 100 inputs. So we have  to multiply it by root 2 over 100. 
And there we go. You can see we are,  in fact, getting some non-zero numbers. 
That's very encouraging, even after going  through 50 layers of depth. So that's good news. 
So this is called climbing.  It's either called Kaiming   initialization or called He initialization. And notice it looks like it's spelled “he”,  
but it's a Chinese surname. So  it's actually pronounced “huh”.  OK, maybe that's why a lot of people  increasingly call it Kaiming initialization. 
I don't have to say his surname, just a  little bit harder to pronounce. All right. 
Applying an init function
So how on earth do we actually use this now  that we know what initialization function   to use for a deep neural network  with a ReLU activation function? 
The trick is to use a method called  apply, which all nn.Modules have. 
So if we grab our model, we can apply any function  we like. For example, let's apply the function. 
Print the name of the type. So here you can  see it's going through and it's printing out  
all of the modules that are inside our model. And notice that our model  
has modules inside modules. It's a Conv in a Sequential in a   Sequential, but model.apply goes through  all of them regardless of their depth. 
So we can apply an init function. So we can  apply the init function, which simply does. 
Random numbers. Normally distributed  random numbers times square root  
of 2 over the number of inputs. That's such an easy thing. It's not  
even worth writing. So that's already  been written, but that's all it does.  It just does that one thing is called  init.kaiming_normal. As we've seen before, 
if there's an underscore at the  end of a PyTorch method name,   that means that it changes something in place. So initinit.kaiming_normal_ will modify this  
weight matrix so that it has been initialized  with normally distributed random numbers based on  
root of 2 divided by the number of inputs. Now you can't do that to a sequential layer  
or a ReLU layer or a flattened layer. So we should check that the module is  
a Conv linear layer. And then we can  just say model.apply the function. 
And so if we do that. And now,  I can use our learning rate  
Learning rate finder and MomentumLearner
finder callbacks that we created earlier. And this time I don't have to worry about…  
actually we can create our own ones  because we don't need to use even   the weird gamma thing anymore. So let's go back and copy that. 
Let's get rid of this gamma equals one point  one. It shouldn't be necessary anymore. 
And we can probably make that 4 now. Oh,  I should have it to recreate the model. 
There we go. Okay, so that's  looking much more sensible.   So at least we've got to a point where the  learning rate finder works. That's a good sign. 
So now when we create our Learner, we're  going to use our MomentumLearner still,   after we get the model, we will apply  init_weights and apply also returns the model. 
So we can actually, this is actually going to  return the model with the initialization applied. 
What’s happening is in each stride-2 convolution?
While I wait, I will answer questions.  Okay, so Fabrizio asks, why do we double the  number of filters in successive convolutions? 
So what's happening is in  each stride-2 convolution. 
These are all stride-2 convolutions. So this is  changing the grid size from 28 by 28 to 14 by 14. 
So it's reducing the size of the  grid by a factor of 4 in total.  So basically, so as we go from 1 to 8  from this one to this one, same deal,  
we're going from 14 by 14 to 7 by 7. So it reduced the grid size by 4,  
we want it to learn something. And if you use, if you give it  
exactly the same kind of number of units or  activations, there's not really, it's not  
really forcing it to learn things as much. So ideally, as we decrease the grid size,   we want to have enough channels that  you end up with a few less activations 
than before it, but not too many less. So if  we double the number of channels, then that   means we've decreased the grid size by a model  of 4, increase the channel count by a model of 2. 
So overall, the number of activations  has decreased by a factor of 2. 
And so that's what we want. We want to  be kind of forcing it to find ways of  
compressing the information intelligently. As it goes down. Also, we kind of want to be  
having a roughly similar amount of compute,  roughly similar amount through the neural net. 
So as we decrease the grid size, we can add  more channels because decreasing the grid  
size decreases the amount of compute. Increasing the channels then gives it   more things to compute. So we're kind of  getting this nice compromise between, yeah, 
between the, kind of, amount of compute  that it's doing, but also giving it   some kind of compression work to do. That's the, kind of, the basic idea. 
Normalizing input matrix
Well, still not able to train. Well,  OK, if we leave it for a while.  
OK, it's not great, but it is  actually starting to train.  That's encouraging. And we got up  to a 70% accuracy so we can see. 
Not surprisingly, we're getting  these spikes and spikes.   And so in the statistics, you can see that. Well, it didn't quite work. We don't have a  
mean of 0. We don't have a standard  deviation of 1, even at the start.  Why is that? Well, it's because  we forgot something critical. 
If you go back to our original point, even when  we had our… let's go to the Kaiming version. 
Even when we had the correctly normalized  matrix that we're multiplying by. 
Well, you also have to have a correctly  normalized input matrix. And we never   did anything to normalize our inputs. So our inputs, actually, if we get the...  
Just get the first x mini batch. I get its mean and standard deviation. It has a   main of 0.28 and a standard deviation of 0.35. So we actually didn't even  
start with a 0, 1 input. And so we started with the mean beneath,  
above 0 and a standard deviation beneath 1. So it was very hard for it. So, using the inner  
helped, at least we're able to train a little bit. But it's not quite what we want. We actually need  
to modify our inputs so they have  a mean of 1 and a standard, sorry, 
A mean of 0 and a standard deviation of 1.  So we could create a callback to do that. 
So a callback, let's create a batch transform  callback. And so we're going to pass in a   function that's going to transform every batch. And so just in the before_batch, we will set  
the batch to be equal to the  function applied to the batch. 
Now, I can note, by the way, we  don't need self.learn.batch here.  Because we can read any.. because  it's one of the four things that we  
kind of proxy down to the Learner automatically. But we do need it on the left hand side   because it's only in the getattr, remember. So be very careful. So I might just leave it  
the same on both sides, just so  that people don't get confused. 
OK, so let's create a function  _norm that subtracts the mean  
and divides by the standard deviation. And so, remember, a batch has an x and a  
y. So it's the x part where we subtract the  mean and divide by the standard deviation.  And so the new batch will be that as the x and  the y will be exactly the same as it was before. 
So let's create an instance of the normalization,  of the BatchTransformCB, which is going to do the  
normalization function. We'll call it _norm,   so we can pass that as an  additional callback to our Learner. 
And now. That's looking a lot  better, so you can see here. 
All we had to do was check that our input  matrix was 0, 1. And… mean, standard  deviation, and all of our weight  
matrices was 0, 1 standard deviation. And we didn't have to use any tricks at   all. It was able to train and got  it to an accuracy of 85 percent. 
85% accuracy
And so if we look at the color_dim and  stats, look at this, it looks beautiful. 
Now this is layer one. This is layer  two, three, four. It's still not perfect. 
I mean, there's some randomness, right? And  we've got what is it like seven or eight layers? 
So that randomness does, kind of, as you  go through the layers by the last one,   it still gets a bit ugly and you can kind  of see it bouncing around here as a result. 
And you can see that also in the  means and standard deviations. 
There's some other reasons this is  happening. We'll see in a moment.   But this is the first time we've really got our  even somewhat deep convolutional model to train. 
And so this is a really exciting step. You  know, we have, from scratch, in a sequence of 11  
notebooks managed to create a real convolutional  neural network that is training properly. 
So I think that's pretty amazing. Now, we don't have to use a callback for this. 
The other thing we could do to modify the input  data, of course, is to use the with_transform  
Using with_transform to modify input data
method from the Hugging Face datasets library. So we could modify our transformi  
to do to subtract the mean and  divide by the standard deviation.  And then recreate our DataLoaders. And if  we now get a batch out of that and check it,  
it's now got, yep, a mean of 0  and the standard deviation of 1.  So we could also do it this way. So generally  speaking, for stuff that needs to, kind of,  
dynamically modify the batch, you can often  do it either in your data processing code,  
or you can do it in a callback. And neither is right or wrong.   They both work well. And you can see  whichever one works best for you. 
ReLU and 0 mean
Okay, now I'm going to show you something amazing. 
Okay, so it's great this is training  well, but when you look at our stats,  
despite what we did with the normalized input and  the normalized, the normalized weight matrices, 
we don't have a mean of 0. And we don't have a  standard deviation of 1, even from the start. 
So why is that? Well, the problem is  
that we were putting our data through a ReLU. And our activation stats are looking at  
the output of those ReLU blocks, because  that's kind of the end of each, you know,   that's the activation of each combination of  matrix multiplication and activation function. 
And since a ReLU removes all of the negative  numbers, it's impossible for the output of a ReLU  
to have a mean of 0, unless,  literally, every single number is 0,   because it has got no negatives. So ReLU, seems to me, to be fundamentally  
incompatible with the idea of a correctly  calibrated bunch of layers in a neural net. 
So I came up with this idea of saying, well,  why don't we take our normal ReLU and have the  
ability to subtract something from it. And so we just take the result of our  
ReLU and subtract, so sub of minus, I mean, I  just, I can write this in a more obvious way,  
it's exactly the same as just - =. Why don't I just do that?   We'll subtract something from our ReLU. That will allow us to pull the whole  
thing down so that the bottom of our ReLU is  underneath the x-axis and it has negatives. 
And that would allow us to have a mean of  0. And while we're there, let's also do  
something that's existed for a while. I didn't come up with this idea,   which is just to do a leaky ReLU,  which is where we say, let's not have  
the negatives be totally flat, just truncated.  But instead, let's just have those  numbers decreased by some constant amount. 
Let me show you what that looks like. So those  two together, I'm going to call general ReLU,   which is where we do this thing called leaky  ReLU, which is where we make it so it's not  
flat under 0, but instead just less sloped. And we also subtract something from it.  
So for example, I've created a little  function here for plotting a function.  So let's plot the general ReLU function  with a leakiness of 0.1. So that will  
mean there's a 0.1 slope underneath  the, under 0, and we'll subtract 0.4. 
And so you can see above 0, it's just a normal y  equals x line, but it's been pushed down by 0.4. 
And then when it's less than zero, it's not flat  anymore, but it's just got a slope of one tenth. 
And so this is now something which if you find  the right amount to subtract for each amount  
of leakiness, you can make a mean of 0. And I actually found that this particular   combination gives us a mean of 0 or thereabouts. So let's now  
Changing the activation function
create a new convolution function where we can  actually change what activation function is used, 
that gives us the ability to change the  activation functions in our neural nets.  Let's change get_model to allow  it to take an activation function  
which is passed into the layers. And while we're there, let's also   make it easy to change the number of filters. So we're going to pass in a list of the number  
of filters in each layer and we will default it  to the numbers in each layer that we've discussed.  And so we're just going to go through, in a list  comprehension, creating a convolution from the  
previous number of filters, this number  of filters to the next number of filters.  And we'll pop that all into a sequential  along with a flatten at the end. 
And while we're there, we also then need to  be careful about init_weights because this  
is something that people tend to forget. Which is that init, it's just that Kaiming  
initialization, the default only applies at all  to layers that have a ReLU activation function. 
We don't have ReLU anymore.  We actually have leaky ReLU. 
The fact that we're subtracting a  bit from it doesn't change things,   but the fact that it's leaky does. Now luckily, a lot of people don't  
know this, but actually PyTorch's Kaiming  normal has an adjustment for leaky ReLUs. 
Weirdly enough, they just call it a. So if you pass into the Kaiming normal   initialization, how your  leaky values, leaky factor  
as a, then you'll get the correct  initialization for a leaky value.  So we need to change init_weights  now to pass in the leakiness. 
All right. So let's put all this together. So, our general ReLU activation function is  
GeneralRelu with a leak of  0.1 and a subtract of 0.4.  So we use partial to create a function  that has those built-in parameters. 
For ActivationStats, we need to update it  now to look for GeneralRelu, not nn.ReLU. 
Okay. And then our init_weights function, we're  going to have a partial with leaky equals 0.1. 
So we'll call that our init weights. 
Great. So now we'll get our model, using that new  activation function and that new init weights. 
And we'll fit that. 
Oh, that's encouraging. Accuracy of 845, which is  about as high as we got to at the end previously. 
87% accuracy and nice looking training graphs
Wow. Look at that. So we're  up to an accuracy of 87  
percent. And let's take a look. Yeah,   I mean, look, we still got a little bit of  a spike, but it's almost smooth and flat. 
And let's have a look here. Look at that. A mean,  it's standing at about 0. Standard deviation. 
Standard deviation is still a bit low, but  it's coming up around 1. It's not too bad.  Generally around 0.8. So it's all  looking pretty encouraging, I think. 
And oh, yeah. Look, the percentage of  dead units in each layer is very small. 
So finally, we've really trained, you know, got  some very nice looking training graphs here. 
And yeah, it's interesting that  we had to literally invent our own  
activation function to make this work. And I think that gives you a sense of  
how few people actually care  about this, which is crazy,   because as you can see, it's in some  ways, it's the only thing that matters. 
And it's not at all mathematically  difficult to make it all work. 
And it's not at all computationally  difficult to see whether it's working.  But other frameworks don't even  let you plot these kinds of things. 
So nobody even knows that they've  completely messed up their initialization. 
So, yeah, now, you know. Now, some very nice news. So the  
first thing to be aware of, which is tricky, is  a lot of models use more complicated activation  
functions nowadays, rather than ReLU or  leaky ReLU or even this general version. 
You need to initialize your neural  network correctly, and most people don't. 
And sometimes nobody's even figured  out or bothered to try to figure out  
what the correct initialization to use is. But there's actually a very cool trick  
“All You Need Is a Good Init”: Layer-wise Sequential Unit Variance
which almost nobody knows about, which is a  paper called “All You Need Is a Good Init”,  
which Dmytro Mishkin wrote a few years ago. And what Dmytro showed is that there's actually  
a completely general way of initializing  any neural network correctly, regardless  
of what activation functions are in it. And it uses a very, very simple idea. 
And the idea is create your model,  initialize it however you like,  
and then go through and put a  single batch of data through.  And look at the first layer, see what the mean  and standard deviation through the first layer is. 
And if the mean, you know, if the  standard deviation is too big,   divide the weight matrix by a bit. If the mean is too high,  
subtract a bit off the weight matrix. And do that repeatedly for the first layer until  
you get the correct mean and standard deviation. And then go to the second layer,   do the same thing. Third layer, do the same thing, and so forth. 
So we can do that using hooks, right? So we could create a little, so this is  
called Layer-wise Sequential Unit Variance, LSUV. We can create a little LSUV stats that will grab  
the mean of the activations of a layer and the  standard deviation of the activations of a layer. 
And we will create a hook with that function. And what it's going to do is after the,  
after we've run that hook, to find out the  mean and standard deviation of the layer, 
we will go through and run the model, get the  standard deviation and mean, see if the standard  
deviation is not 1, see if the mean is not 0. And we will subtract the mean from the bias  
and we will divide the weight  matrix by the standard deviation. 
And we will keep doing that until we get  a standard deviation of 1 and a mean of 0. 
And so by making that a hook,   what we will do is we will grab all  the ReLU's and all the convs, right? 
And so just to show you what happens  there, once I've got all the ReLU's   and all the convs, I can use zip. So zip in Python takes a bunch of lists  
and creates a list of the items, the first items,  the second items, the third items and so forth. 
So if I go through the zip of ReLU's and  convs and just print them out, you can   see it prints out the ReLU and the first conv. The second ReLU, the second conv, the second ReLU,  
sorry, the third ReLU, the  third conv and so forth.  We use zip all the time in Python. So it's a really important thing to be aware of. 
So we could go through the ReLU's and the convs  and call lsuv_init, passing in those module pairs. 
Sorry, passing in, yeah, passing  in the ReLU and the conv.  And then for each one, oh, and  we're going to do that on the batch. 
And of course, we need to put the batch  on the correct device for our model. 
And so now that I've done that,  
we now have, it ran almost instantly. It's now made all the biases and  
weights correct, give us 0, 1. And now if I train it, there it is. 
So we didn't do any initialization at all  of the model other than just call lsuv_init. 
And this time we've got an accuracy of 0.86  
versus previously it's 0.87. So pretty much the same thing, close enough. 
And actually, if you want to actually  see that happening, I guess what we  
could do, it's going to be pretty obvious, after we run this, we could say print(h.mean,  
h.std). Actually,  
we could do it before and afterwards, right? So we could say, right, before and after. 
There we go.  Yeah, so it starts at, so the first layer started  at a mean of -0.13 and a variance of 0.46. 
And it kept doing the divide,  subtract, divide, subtract,   divide, subtract until eventually it got  to mean of 0, standard deviation of 1. 
And then it went to the next layer and it  kept going, going, going until that was 0, 1. 
And then the third layer  and then the fourth layer.  And so at that point, all of the layers had  a mean of 0 and a standard deviation of 1. 
So I guess. like, one thing with LSUV, you know,  it's kind of very mathematically convenient. 
We don't have to spend any  time thinking about, you know,   if we've invented a new activation function  or we're using some activation function where  
nobody seems to have figured out the correct  initialization for it, we can just use LSUV. 
It did require a little bit more fiddling  around with hooks and stuff to get it to work.  And I haven't even put this into,  like, a callback or anything. 
So if you yeah, if you decide you  want to try using this in some of   your models, it might be a good idea. And it actually be good homework to  
see if you can come up with a callback  that does LSUV initialization for you. 
That would be pretty cool, wouldn't  it? In before_fit, I guess it would be. 
You'd have to be a bit careful  because if you ran fit multiple times,   it would actually initialize it each time. That would be one issue with that to think about. 
Batch Normalization, Intro
OK, so something which is quite  similar to LSUV is batch normalization. 
So we're going to have a seven minute break and  then we're going to come back and we're going to  
talk about batch normalization. See you in seven minutes. 
OK, hi, let's do this, Batch Normalization. Batch Normalization was such an important paper. 
I remember when it came out, I was at Enlitic,  my medical startup, and I… think that's right. 
And everybody was talking about it. And in particular,  
they were talking about this graph that basically  showed what it used to be like until batch norm to  
train a model on ImageNet. How many  
training steps you'd have to do  to get to a certain accuracy. 
And then they showed what  you could do with batch-norm. 
So much faster. It was amazing. And we all  thought that can't be true, but it was true. 
So basically, the key idea of batch-norm  is that with LSUV and input normalization  
and Kaiming init, we are normalizing the  layers, each layer's inputs before training. 
But the distribution of each layer's  inputs changes during training. 
And that's a problem.  So you end up having to  decrease your learning rates. 
And as we've said, you have to be very  careful about parameter initialization. 
So the fact that the layers inputs change during  training, they call internal covariate shift,  
which for some reason, a lot of people tend to  find a confusing statement or confusing name,   but it's very clear to me. And you can fix it by  
normalizing layer inputs during training. So you're making the normalization a part  
of the model architecture, and you perform  the normalization for each mini batch.  Now, I'm actually not going to  start with batch normalization. 
Layer Normalization
I'm going to start with something that came  out one year later called layer normalization,   because layer normalization is simpler. Let's do the simpler one first. 
So Layer Normalization came out as a…  
this group of fellows, the last  of whom I'm sure you've heard of.  And it's probably easiest to  explain by showing you the code. 
So if you're thinking, “Layer Normalization”,  wow, it's a whole paper, a Geoffrey  
Hinton paper must be complicated. No, the whole thing is this code.  What is layer normalization? Well, we can create a module. 
And we're going to pass in. 
We don't need to pass in anything, actually,  you can totally ignore the parameters for now.  In fact, what we're going to do is we're  going to have a single number called mult,  
for the multiplier and a single number called add. That's the thing we're going to add. And we're  
going to start off by multiplying  things by 1 and adding 0.  So we're going to start off  by doing nothing at all. 
Okay, this is the layer.  It has a forward function.  And in the forward function, so  remember that, by default, we have NCHW. 
We have batch by channel by height by width. We're going to take the mean  
over the channel, height and width. So we're just going to find the mean activation  
for each input in the mini batch. And when I say input, though, remember that  
this is going to be, this is a layer, right? So we can put this layer any way we like.  So it's the input to that layer. And we'll  do the same thing for finding the variance. 
Okay. And then we're going to normalize  our data by subtracting the mean  
and dividing by the square root of the variance,  which of course is the standard deviation. 
We're going to add a very small number,  by default 1e-5 to the denominator. 
Just in case the variance is 0 or ridiculously  small, this will keep the number from going giant. 
Just if we happen to get something  with a very small variance.  This idea of an epsilon as being something  we add to a divisor is really, really common. 
And in general, you should not  assume that the defaults are correct.  Very often the defaults are too small  for algorithms that use an epsilon. 
Okay. So here we are, as you can  see, we are normalizing the batch. 
I mean, I can call it a batch, but just remember,  it isn't necessarily the first layer, right? 
So it's wherever, whichever  layer we decide to put this in.  So we normalize it. Now the thing is,  
maybe, we don't want it to be normalized. Maybe we want it to have something other than  
a unit variance and something  other than zero mean.  Well, what we do is we then multiply  it back by self.mult and add self.add. 
Now remember self.mult was 1 and self.add is 0. So at first that does nothing at all. 
So at first this is just normalizing the data. So that's good. 
But because these are parameters,  these two numbers are learnable.  That means that the SGD algorithm can change them. So there's a very subtle thing going on here,  
which is that in fact, this might  not be normalizing the data at all,  or normalizing the inputs to  the next layer at all, because  
self.mult and self.add could be anything. So I tend to think that when people think  
about these kind of things like layer  normalization and batch normalization,  thinking of this as normalization in some  ways is not the right way to think of it. 
It's actually doing something I think  to really… well, it's definitely   normalizing it for the initial layers. And we don't really need LSUV anymore  
if we have this in here, because it's  going to normalize it automatically.  So that's handy. But after a few batches,  
it's not really normalizing at all. But what it is doing is, previously,  
this idea of like, how big are the numbers overall  and how much variation do they have overall, 
was kind of built into every single number  in the weight matrix and in the bias vector. 
This way, those two things have  been turned into just two numbers. 
And I think this makes training  a lot easier for it, basically,   to just have just two numbers that it can focus on  to change this overall positioning and variation. 
So there's something very subtle going  on here, because it's not just doing   normalization, at least not after  the first few batches are complete, 
because it can learn to create any  distribution of outputs it wants. 
So there's our layer. So we're going to need   to change our conv function again. Previously, we changed it to add  
activation function to be modifiable. Now we're going to also change it to allow  
us to add normalization layers to the end. So our basic layers, well, we start  
off by adding our Conv2d, as usual. And then, if you're doing normalization,  
we will append the normalization  layer with this many inputs. 
Now, in fact, LayerNorm doesn't  care how many inputs, so I just   ignore it, but you'll see BatchNorm care. If you've got an activation function, add it. 
And so our convolutional layer is  actually a sequential bunch of layers. 
Now, one thing that's interesting, I think,  is that for bias in the conv, if you're using,  
well, this isn't quite true, is it? I was going to say if you're using LayerNorm,   you don't need bias, but actually you kind of do. 
So maybe we should actually change that. For BatchNorm, we won't need bias,  
but actually for this one we do. So let me put this back. bias=True. 
bias=bias. OK. 
So then these initial layers are here. So they all have bias. 
And then we've got bias=false  (OOPS, it should be True). 
OK. So now in our model, we're going to  add layer normalization to every layer  
except for the last one. And let's see how we go. 
Oh, nice 873. OK. 860 and 872. 
So just, we've just got our best by a little bit. So that's cool. 
So the thing about these  normalization layers is, though, that  
they do cause a lot of challenges in models. And generally speaking, ever since batch norm  
appeared, well, there's been this kind  of like big change of view towards it. 
At first people were like, oh,  my God, batch norm is our savior.  And it kind of was, it let us train much deeper  models and get great results and train quickly. 
But then increasingly people realized  it also added a lot of complexity. 
These learnable parameters turned  out to create all kind of complexity.  And in particular batch norm, which we'll see  in a minute, created all kinds of complexity. 
So there has been a tendency in  recent years to be trying to get   rid of or at least reduce the  use of these kinds of layers. 
So, knowing how to actually initialize your models  correctly, at first, is becoming increasingly  
important as people are trying to move away  from these normalization layers increasingly. 
So I will say that. So they're still very helpful,   but they're not a silver bullet, as it turns out. All right. So now let's look at BatchNorm. 
Batch Normalization
So BatchNorm is still not huge, but  it's a little bit bigger than LayerNorm. 
And you'll see that we've now, we've  got the mult and add as before. 
But it's not just one number to add or  one number to multiply, but actually  
we've got a whole bunch of them. And the reason is that we're   going to have one for every channel. And so now when we take the mean and the  
variance, we're actually taking it over the batch  dimension and the height and width dimensions. 
So we're ending up with one mean per  channel and one variance per channel. 
So just like before, once we  get our means and variances,  
we subtract them out and divide them  by the epsilon modified variance. 
And just like before, we then  multiply by mult and add add.  But now we're actually multiplying by a vector  of mults and we're adding a vector of adds. 
And that's why we have to  pass in the number of filters,   because we have to know how many ones and how  many zeros we have in our initial molts and adds. 
So that's the main difference in a sense is that  we have one per channel and that we're also taking  
the average across all of the things in the batch. Whereas in LayerNorm, we didn't. 
Each thing in the batch had its own  separate normalization it was doing. 
Then there's something else in BatchNorm, which  is a bit tricky, which is that during training,  
we are not just subtracting the mean and  the variance, but instead we're getting  
an exponentially weighted moving average of the  means and the variances of the last few batches. 
That's what this is doing. So we start out, so we   basically create something called  vars and something called means. 
And initially the variances are  all 1 and the means are all 0.  And there's one per channel just  like before or one per filter. 
This is number of filters. Same idea, I guess.  Filters we tend to actually use inside the model  and channels we tend to use as the first input. 
So I should probably say filters. Either works though. 
So we get our, let's for example,  we get our mean per filter.  And then what we do is we  use this thing called lerp. 
And lerp is simply saying,  yes, that's what it's done. 
So what lerp does is it takes two numbers, in this  case, I'm going to take 5 and 15 or two tensors. 
They could be vectors or matrices, and  it creates a weighted average of them.  And the amount of weight it  uses is this number here. 
Let me explain. In this case, if I put 0.5, it's going to take   half of this number plus half of this number. So we end up with just the mean. 
But what if we used 0.75?  Then that's going to take 0.75 times  this number plus 0.25 of this number. 
So it basically kind of allows  it to be on like a sliding scale.  So one extreme would be to take all of the second  number, so that would be lerp with 1 there. 
And the other extreme would  be all of the first number.  And then you can slide  anywhere between them, like so. 
So that's exactly the same as saying  5 times 0.9 plus 15 times 0.1. 
So this number here is how much  of the second number do we have. 
And 1 minus that is how much  of this number do we have.  And you can also move this, as you can  with most PyTorch things, you can move  
the first parameter into there. And get exactly the same result. 
So that's what lerp is. So what we're doing here  
is we're doing an in-place lerp. So we're replacing self.means with  
1 minus momentum of self.means. And plus self.momentum times  
this particular mini batches mean. So this is basically doing Momentum again,  
which is why we indeed are calling  the parameter mom for momentum. 
So with a mom of 0.1, which I kind of think is  the opposite of what I'd expect momentum to mean,  
I'd expect it to be 0.9. But with a mom of 0.1,   it's saying that each mini batch self.means will  be 0.1 of this particular mini batches mean. 
And 0.9 of the previous one. The previous sequence, in fact. 
And that ends up giving us what's called  an exponentially weighted moving average. 
And we do the same thing for variances. Okay, so that's only updated during training. 
And then during inference, we just  use the saved means and variances. 
So this, and then why do we have  buffers? What does that mean?  These buffers mean that these means and variances  will be actually saved as part of the model. 
So it's important to understand that this  information about the means and variances  
that your model saw are saved in the model. And this is the key thing which makes  
batch norm very tricky to deal  with and particularly tricky,   as we'll see in later lessons  with transfer learning. 
But what this does do is that it means that we're  going to get something that's much smoother.  You know, a single weird mini batch  shouldn't screw things around too much. 
And because we're averaging across the mini  batch, it's also going to make things smoother.  So this whole thing should lead  to a pretty nice, smooth training. 
So we can train this. So we're going to, this time,   we're going to use our BatchNorm layer for norm. Oh, actually, we need to put the bias thing. 
Is that right? Oh, no, it's no, that's fine. 
Okay.  And one interesting thing I found here is I was  able to now finally increase the learning rate  
up to 0.4 for the first time. So each time I was really trying   to see if I can push the learning rate. And I'm now able to double the learning rate  
and still, as you can see, it's training  very smoothly, which is really cool. 
So there's actually a number of different types  of normal- layer based normalization we can use. 
Batch Norm, Layer Norm, Instance Norm and Group Norm
In this lesson, we've specifically  seen Batch Norm and Layer Norm.  I wanted to mention that there's  also Instance Norm and Group Norm. 
And this picture from the Group  Norm paper explains what happens.  What it's showing is that we've got here the NCHW. And so they've kind of concatenated flattened HW  
into a single axis since they can't draw 4D cubes. And what they're saying is in Batch Norm,  
all this blue stuff is what we average over. So we average across the batch and across  
the height and width. And we end up with one,   therefore, normalization number per channel. So you can kind of slide these blue blocks across. 
So Batch Norm is averaging over  the batch and height and width.  Layer Norm, as we learned, averages over  the channel and the height and the width. 
And it has a separate one  per item in the mini batch. 
I mean, kind of. It's a bit subtle, right? Because, remember, the overall mult and add,  
it just had literally a  single number for each, right?  So it's not quite as simple as  this, but that's a general idea. 
Instance Norm, which we're not looking at  today, only averages across height and width. 
So there's going to be a separate one for every  channel and every element of the mini batch. 
And then finally, Group Norm, which I'm  quite fond of, is like Instance Norm, 
but it arbitrarily basically groups  a bunch of channels together.  And you can decide how many groups of  channels there are and averages over them. 
Group Norm tends to be a bit slow,  unfortunately, because the way   these things are implemented is a bit tricky. But Group Norm does allow you to, yeah, avoid some  
of the challenges of some of the other methods. So it's worth trying if you can. 
And of course, Batch Norm has the additional  thing of the kind of momentum-based statistics. 
But in general, the idea of like, do  you use momentum-based statistics? 
Do you store things per channel or a single  mean and variance in your buffers or whatever? 
You know, all that kind of stuff,  along with what do you average over?  They're all somewhat independent choices you can  make, and particular combinations of those have  
been given particular names. And so there we go.  OK, so we're getting, you know, we've got  some good initialization methods here. 
Putting all together: Towards 90
Let's try putting them all together.  And one other thing we can do is we've been  using a batch size of 1024 for speed purposes. 
If we drop it down a bit to 256,  it's going to mean that it's going   to get to see more mini batches. So that should improve performance. 
And so we're trying to get to 90%, remember? So let's do all this. 
This time we'll use, PyTorch  has its own BatchNorm.  We'll just use PyTorch’s. There's nothing wrong with ours,   but we try to switch to PyTorch’s when  something we've recreated exists there. 
We'll use our MomentumLearner. And we'll fit for 3 epochs. 
And so as you can see, it's going  a little bit more slowly now. 
And then the other thing I'm going  to do is, I'm going to decrease the  
learning rate and keep the existing model. And then train for a little bit longer. 
The idea being that, as the, you know, as it's  kind of getting close to a pretty good answer, 
maybe it just wants to be able  to fine tune that a little bit.  And so by decreasing the learning rate, we  give it a chance to fine tune a little bit. 
So let's see, how are we going? So we got to 87.8% accuracy after three epochs,  
which is an improvement,  I guess, mainly thanks to,  well, basically thanks to using  this smaller mini batch size. 
Now with a smaller mini batch size, you  do have to decrease the learning rate.  So I found I could still get away  with 0.2, which is pretty cool. 
And look at this after just one more  epoch by decreasing the learning rate,   we've got up to 89.7. Oh, we didn't make it. 89.9. 
So towards 90%, but not quite 90%, 89.9. So we're going to have to do some more work  
to get up to our magical 90% number. But we are getting pretty close. 
All right. So that is   the end of initialization, an incredibly  important topic, as hopefully you've seen. 
Accelerated SGD
Accelerated SGD.  Let's see if we can use this to  get us up to 90 plus, above 90%. 
So let's do our normal imports  and data setup as usual.  And so just to summarize what  we've got, we've got our MetricsCB. 
We've got our ActivationStats on the GeneralRelu. So callbacks are going to be the DeviceCB,  
put it on CUDA or whatever, the metrics,  the progress bar, the activation stats. 
Our activation function is going  to be a GeneralRelu with 0.1   leakiness and 0.4 subtraction. And we've got the init_weights,  
which we need to tell it about how leaky they are. And then if we're doing a learning rate finder,  
we've got a different set of callbacks. So it's no real reason to have a progress bar  
callback with a learning rate finder, I guess. It's pretty short anyway.  Oh, which reminds me, there was one little  thing I didn't mention in initializing,  
which is a fun trick you might  want to play around with.  And in fact, Sam Watkins asked a question  earlier in the chat and I didn't answer  
it because it's actually exactly here. In GeneralRelu, I added a second thing  
you might have seen, which is the maximum value. And if the maximum value is set, then I clamp  
the value to be no more than the maximum. So basically, as a result, let's say you set  
it to 3, then the line would go up to here like it  does here, and then it would go up to three like  
it does here, and then it would be flat. And using that can be a nice way. 
I mean, that probably got higher  up to about six, but that can be   a nice way to avoid numbers getting too big. And maybe if you really wanted to have fun,  
you could do kind of like a leaky maximum, which  I haven't tried yet, where maybe at the top it   kind of goes like, you know, 10 times smaller,  kind of just exactly like the leaky could be. 
So anyway, if you do that,  you'd need to make sure that   the, you know, that you're still getting  0, 1 layers with your initialization. 
But that would be something you  could consider playing with. 
Okay, so let's create our own little SGD class. So an SGD class is going to need to know  
what parameters to optimize. And if you remember, the module dot parameters  
method returns a generator. So we use a list to turn, you know,  
we want to turn that into a list. So it's  kind of forced to be a particular, you know,   not something that's going to change. We're going to need to know the learning  
rate. We're going to need to know the weight  decay, which we'll look at a bit in a moment. 
And for reasons we'll discuss later, we also want  to keep track of what batch number are we up to. 
So an optimizer basically has two  things, a step and a zero_grad.  So what steps going to do is, obviously,  with no_grad, because this is not part  
of the thing that we're optimizing. This is the optimization itself. We   go through each tensor of parameters  and we do a step of the optimizer. 
And we'll come back to this in a moment. We do  a step of the regularizer and we keep track of   what batch number we're up to. And so what does SGD do  
in a step of the optimizer? It subtracts out from the parameter  
its gradient times the learning rate. So that's an SGD optimization step.  
And to zero the gradients, we go  through each parameter and we zero it. 
And that's in torch.no_grad. So I guess it's not. So use .data that way. If you use .data, then you  
don't need to say the no_grad. Just a little typing saver.  
OK, so let's create a TrainLearner. So it's a Learner with a training callback kind of  
built in and we're going to set the optimization  function to be this SGD we just wrote.  And we'll use the BatchNorm model with the  weight initialization we've used before. 
And if we train it, then just this is just to give  us basically the same results we've had before. 
While this is training, I'm going  to talk about regularization. 
Regularization
Hopefully you remember from Part 1 of this course  or from your other learning what weight decay is. 
And so just to remind you, weight decay or  L2 regularization are kind of the same thing. 
And basically what we're doing is we're  saying let's add the square of the weights  
to the loss function. Now, if we add the square of the  
weights to the loss function, so. Whatever our loss function is. 
So we'll just call it loss, bop,  bop, bop, bop. We're adding plus  
the sum of the square of the weights. So that's our L.   And so the only thing we actually  care about is the derivative of that. 
And the derivative of that is  equal to the derivative of the loss  
plus the derivative of this,  which is just the sum of 2w. 
And then what we do is we multiply this bit here  by some constant, which is the weight decay. 
So we call that weight decay. And so  since the weight decay could directly   incorporate the number, the 2, we can  actually just delete that entirely. 
And just. 
Time weight decay to that  and doing this very quickly   because we have already covered it in Part 1. 
So this is hopefully something that  you've all seen before. So we can. 
Do weight decay by taking our gradients  and adding on the weight decay times 
the weights. 
And so as a result, then in SGD,  because that's part of the gradient. 
Oh, man, I got it the wrong way around. Need to do.  That first. I guess it does. Well, whatever. OK. 
So since that's part of the gradient, then  in the optimization step, that's using  
the gradient and it's subtracting  out gradient times learning rate. 
But what you could do is because we're  just ending up doing p.grad * self.lr. 
And the p.grad update is  just to add in wd * weight. 
We could simply skip updating the gradients  and instead directly update the weights. 
To subtract out the learning  rate times the wd times weight.  So they would be mathematically  identical. And that is what  
we've done here in the regularization step. We basically say if you've got weight decay. 
Then. Just take p *= 1 - lr * wd,  
which is mathematically the same as this. Because we've got weight on both sides. 
So that's why the regularization  is here inside our SGD. 
And yes, that's finished running. That's good.  We've got an 85% accuracy that all looks fine. 
Momentum
And we're able to train it a high  learning rate of 0.4. That's pretty cool.  So now let's add momentum. Now, we had a kind of a  hacky momentum learner before, but we're actually,  
momentum should be in an optimizer, really. And so let's talk a bit about   what momentum actually is. So let's just create some data. 
So our xs is just going to be  equally spaced numbers from -4 to 4.  A hundred of them. And our  ys is just going to be xs 
divided by 3, squared, 1 minus  that, plus some randomization. 
And so these dots here is our random data. I'm going to show you what momentum is by example.  
And this is something that  Sylvain Gugger helped build.  So thank you, Sylvain, for our book,  actually, if memory serves correctly. 
Actually, it might even be the course before  that. What we're going to do is we're going   to show you what momentum looks like for  a range of different levels of momentum. 
These are the different levels we're  going to use. So let's take a beta of 0.5.  So that's going to be our first one. So we're  going to do a scatter plot of our xs and ys. 
It's the blue dots. And then we're going to go  through each of the ys and we're going to do, 
this hopefully looks familiar. This is  doing a lerp. We're going to take our  
previous average, which will start at 0. Times beta, which is 0.5 plus 1 minus  
beta. That's 0.5 times our new average. And then we'll append that to this red line. 
And we'll do that for all the data points and then  plot them. And you can see what happens when we  
do that is that the red line becomes less bumpy. Right?. Because each one is half this exact dot  
and half of whatever the red line previously was. So again, this is an exponentially weighted  
moving average. And so we could  have implemented this using lerp. 
So as the beta gets higher, it's saying: do more  of just be wherever the red line used to be and  
less of where this particular data point is. And so that means when we have these kind of   outliers, the red line doesn't  jump around as much, as you see. 
But if your momentum gets too high,  it doesn't follow what's going on at  
all. And in fact, it's way behind. Right?. When you're using momentum,   it's always going to be partially responding  to how things were many batches ago. 
And so even at beta of 0.9 here. The red line is offset to the right because,  
again, it's taking it a while for it to recognize  that all things have changed because each time  
it's 0.9 of it is where the red line used to be. And only 0.1 of it is what is this data point say.  
So that's what momentum does. So the reason that momentum is useful  
is because when you have a loss function  that's actually kind of like very, very bumpy. 
Like that, right. You want to be able to follow the actual curve, right?. So using momentum,  
you don't quite get that, but you kind of  kind of a version of that that's offset.  To the right a little bit, but still, you  know, hopefully spending a lot more time, 
you don't really want to be heading off  in this direction, which you would if you   follow the line and then this direction,  which you would if you follow the line. 
You really want to be following  the average of those directions.   And that's what momentum lets you do. 
So to use momentum,  we will inherit from SGD and we will override the  definition of the optimization step. Remember,  
there was two things that step called it called  the regularization step and the optimization step. 
So we're going to modify the  optimization step. We're not   just going to do minus equals grad times self.lr. 
But instead, then when we  create a momentum object. 
We will tell it what momentum  we want for default point nine.  Store that away. And then  in the optimization step. 
For each parameter, because remember,  the optimization step is being called   for each parameter in a model. So that's each layer's weights  
and each layer's biases, for example. We'll find out for that parameter. Have   we ever stored away its moving  average of gradients before? 
And if we haven't, then we'll set them  to 0 initially, just like we did here. 
And then. We will do our look, right?. So we're going to say the moving average of,  
exponentially weighted moving average of  gradients is equal to whatever it used to be… 
times the momentum plus this actual new batches  
gradients times one minus momentum. So that's just doing the lerp,   as we discussed. And so then we're just going  to do exactly the same as the SGD update step. 
But instead of multiplying by p.grad,  we're multiplying it by p.grad_avg.  So there's a cool little trick here, right? Which  is that we are basically inventing a brand new  
attribute, putting it inside the parameter tensor. And that attribute is where we're storing away  
the moving average, exponentially  weighted moving average of   gradients for that particular parameter. So as we look through the parameters, we don't  
have to do any special work to get access to that. So I think that's pretty handy.  
All right. So one interesting thing,   very interesting here I found is I could really  hike the learning rate way up to one point five. 
And the reason why is because we're not  getting these huge bumps anymore. And   so by getting rid of the huge bumps, the  whole thing's just a whole lot smoother. 
So previously we got up to 85%. Because we've gone back to our 1024 batch size  
and just 3 epochs and a constant learning rate. And look at that. We've gone up to 87.6%. 
So it's really improved things. And the loss  function is nice and smooth, as you can see. 
OK. And so then in our color_dim plot, you  can see, this is actually this really the  
really the smoothest we've seen. And it's a bit different to the   momentum learner because the momentum  learner didn't have this one minus part. 
It wasn't lerping. It was, it was basically  always including all of the grad plus a  
bit of the momentum part. So this is yeah, this is a  
different, better approach, I think.  And yeah, we've got a really nice, smooth result. One person's asking, “don't we get a similar  
Batch size
effect, I think, in terms of the smoothness  if we increase the batch size?”, which we do.  But if you just increase the batch size,  you're giving it less opportunities to update. 
So having a really big batch  size is actually not great.  Yann LeCun, who created the first really  successful ConvNets, including LeNet-5,  
says he thinks the ideal batch size,  if you can get away with it, is 1.  But it's just slow. You want it to have as  many opportunities to update as possible. 
There's this weird thing recently  where people seem to be trying to   create really large batch sizes, which  to me is, yeah, doesn't make any sense. 
We want the smallest batch size we  can get away with, generally speaking,   to give it the most chances to update. So this has done a great job of that.  
And we're getting very good results despite  using only 3 epochs of very large batch size. 
Okay, so that's called Momentum. Now, something that was developed   in a course or announced in a  Coursera course back in maybe 2012,  
RMSProp
2013 by Geoffrey Hinton, has never  been published, is called RMSProp. 
Let's have it running while we talk about it. RMSProp is going to update the optimization  
step using something very similar to  Momentum, but rather than lerping on the  
p.grad, we're going to lerp on p.grad squared. And just to keep it kind of consistent,  
we won't call it mom, we'll call it  sqr_mom, but this is just the multiplier.  And what are we doing with the grad squared? Well, the idea is that a large grad squared  
indicates a large variance of gradients. So what we're then going to do is divide by  
the square root of that plus epsilon. Now, you'll see I've actually been a bit  
all over the place here with my batch norm. I put the epsilon inside the square root. 
In this case, I'm putting the  epsilon outside the square root.  It does make a difference. And so be careful  as to how your epsilon is being interpreted. 
Generally speaking, I can't  remember if I've been exactly right,   but I've tried to be consistent with  the papers or normal implementations.  This is a very common cause of  confusion and errors, though. 
So what we're doing here is we're dividing  the gradient by the amount of variation. 
So the square root of the moving  average of gradient squared.  And so the idea here is that if the gradient  has been moving around all over the place,  
then we don't really know what it is. Right?. So we shouldn't do a very big update  
if the gradient is very, very  much the same all the time. 
Then we're very confident about it.  So we do want to be a big update.  I have no idea why we're doing this in  two steps. Let's just pop this over here. 
Now, because we are dividing our gradient by  this generally possibly rather small number,  
we generally have to decrease the learning rate. So bring the learning rate back to   point one. And as you see, it's training. It's not amazing, but it's training OK. 
So RMSProp can be quite nice. It’s a bit bumpy there, isn't it? I mean,  
I could try decreasing it a little bit. Maybe down to 3e-3 instead. 
That's a bit better. And a bit smoother. That's probably good. Let's see what the  
colorful dimension plot looks like, shall we?. We know again, it's very nice,   isn't it? That's great. Now, one thing I did,  
which I don't think I've seen done before,  I don't remember people talking about,  is I actually decided not to do  the normal thing of initializing  
to zeros, because if I initialize to zeros, then my initial denominator here will basically  
be zero plus epsilon, which will mean  my initial learning rate will be very,  very high, which I certainly don't  want. So I actually initialized it,  
at first, to just whatever the first  mini batches gradient is, squared. 
And I think this is a really useful  little trick for using RMSProp. 
Momentum can be a bit aggressive  sometimes for some really  
finicky learning methods, finicky architectures. And so RMSProp can be a good way to get reasonably  
fast optimization of a very finicky architectures. And in particular, EfficientNet is an architecture  
which people have generally  trained best with RMSProp.  So you don't see it a whole lot,  but you know, in some ways it's  
just historical interest, but you see it a bit. But I mean, the thing we really want to look at   is RMSProp plus Momentum together and  RMSProp plus Momentum together exists. 
Adam: RMSProp plus Momentum
It has a name. You will have heard the name  many times. Name is Adam. Adam is literally  
just RMSProp and Momentum. So we, rather annoyingly,  
call them beta1 and beta2. They should be called momentum and   square momentum or momentum of squares, I suppose. So beta1 is just the momentum from the momentum  
optimizer. beta2 is just these momentum  for the squares from the RMSProp optimizer. 
So we'll store those away. And just  like RMSProp, we need the epsilon. 
So I'm going to, as before, store away the  gradient average and the square average. 
And then we're going to do our lerping.   But there's a nice little trick here,  which is in order to avoid doing this, 
where we just put the initial batch  gradients as our starting values,  
we're going to use zeros as our starting values. And then we're going to unbiased them. 
So basically, the idea is that for the  very first mini batch, if you have zero  
here being lerped with the gradient, then the first mini batch will obviously  
be closer to zero than it should be. But we know exactly how much closer it  
should be to zero, which is just it's  going to be self.beta1 times closer, 
at least in the first mini batch, because that's  what we've lerped with. And then the second mini   batch will be self.beta1 squared. And so in the third mini batch  
to be self.beta1 cubed and so forth. And that's why we had this self.i back in our SGD,  
which was keeping track of  what mini batch we’re up to.  So we need that in order to do  this unbiasing of the average. 
Oh, dear, I'm not unbiasing  the square of the average.  Am I? No, I'm not. Oops. So we need to do that here as well. I wonder  
if this is going to help things a little bit. unbias_sqr_avg is going to be p.sqr_avg. 
And that will be beta2. And so we  will use those unbiased versions. 
So this unbiasing only matters  for the first few mini batches,   where otherwise it would be too close to zero, you know, be closer to zero than it should be.  
Right. So we run that.  
And so, again, you know, we've,  you would expect the learning   rate to be similar to what RMSProp needs  because we're doing that same division. 
So we actually do have the  same learning rate here.  And yeah, so we're up to 86.5% accuracy. So that's pretty good. I think. 
Yeah, it's actually a bit less  good than Momentum, which is fine.   You know, obviously you can fiddle around. Well, Momentum we had  
0.9. Yeah. So you can fiddle around  with different values of beta2, beta1. 
See if you can beat the Momentum  version. I suspect you probably can. 
OK. We're a bit out of time, aren't we? All right. I am excited about the next bit, but I want  
to spend time doing it properly  so I won't rush through it now.  But instead, we're going to do it  next time. So I will. Yes, I will. 
Give you a hint that in our next  lesson, we will, in fact, get above 90%. 
And it's got some very cool stuff to show  you. I can't wait to show you that then. 
But, you know, I think in the meantime, let's  give ourselves a pat on the back that we have  
successfully implemented. You know, I mean, think about all this   stuff we've got running and happening and we've  done the whole thing from scratch using nothing  
but what's in the Python standard library. We've re implemented everything   and we understand exactly what's going on. So I think this is really quite terrifically cool. 
Personally, I hope you feel the same way and  look forward to seeing you in the next lesson. 
Thanks. Bye.

---

# 18

Hi folks, thanks for joining me for Lesson 18. We're going to start today in Microsoft Excel. 
You'll see there's an Excel folder  actually in the course22p2 repo. 
And in there there's a spreadsheet  called graddesc as in gradient descent,  
which I guess we should zoom in a bit here.  So there's some instructions here,  
but this is basically  describing what's in each sheet.  We're going to be looking at the various  SGD accelerated approaches we saw last time,  
but done in a spreadsheet.  We're going to do something very, very simple,  which is to try to solve a linear regression. 
So the actual data was generated with y =  ax + b, where a, which is the slope, was 
And so you can see we've got  some random numbers here. 
And then over here, we've  got the ax + b calculation. 
So then what I did is I copied and pasted as  values, just one set of those random numbers 
into the next sheet called basic. This is the basic SGD sheet.  So that's what x and y are. And so the idea is we're going to try to  
Basic SGD
use SGD to learn that the intercept is 30 and the slope is 2. 
So the way we do SGD is we, so those  are, those are our weights or parameters. 
So the way we do SGD is we start  out at some random kind of guess.  So my random guess is going to be 1  and 1 for the intercept and slope. 
And so if we look at the very first data point,  which is x is 14 and y is 58, the intercept 
and slope are both 1, then  we can make a prediction.  And so our prediction is just equal  to slope times x plus the intercept. 
So the prediction will be 15. Now, actually, the answer was 58. 
So we're a long way off. So we're going to use mean squared error.  So the mean squared error is just the  error, so the difference, squared. 
Okay, so one way to calculate how much would  the prediction, sorry, how much would the 
error change, so how much would the squared  error, I should say, change if we changed 
the intercept, which is b, would be just to  change b by a little bit, change the intercept 
by a little bit and see what the error is. So here that's what I've done is I've just   added 0.01 to the intercept and then calculated y and then calculated the difference squared. 
And so this is what I mean by er b1. This is the error squared I  
get if I change b by 0.01. So it's made the error go down a little bit.  So that suggests that we should probably  increase b, increase the intercept. 
So we can now calculate the estimated derivative  by simply taking the change from when we use 
the actual intercept using  the intercept plus 0.01.  So that's the rise and we divide it by  the run, which is, as we said, is 0.01. 
And that gives us the estimated derivative  of the squared error with respect to b, the 
intercept. Okay, so it's about negative 86, 85.99. 
So we can do exactly the same thing for a,  so change the slope by 0.01, calculate y, 
calculate the difference and square it. And we can calculate the estimated derivative  
in the same way, rise, which is the difference divided by run, which is 0.01. 
And that's quite a big number, minus 1,200. In both cases, the estimated derivatives  
are negative. So that suggests   we should increase the intercept and the slope. And we know that that's true because actually the  
intercept and the slope are both bigger than one.  The intercept is 30, should be  30 and the slope should be 2. 
So there's one way to calculate the derivatives. Another way is analytically. 
And the derivative of squared is 2 times. So here it is here. 
I've just written it down for you. So here's the analytic derivative. 
It's just two times the difference. And then the derivative for the slope is here. 
And you can see that the estimated version  using the rise over run and the little 0.01 
change and the actual, they're pretty similar. Okay.  And same thing here, they're pretty similar. So anytime I calculate gradients  
kind of analytically, but by hand,  I always like to test them against  doing the actual rise over run  calculation with some small number. 
And this is called using the  finite differencing approach.  We only use it for testing because it's slow  because you have to do a separate calculation 
for every single weight. But it's good for testing. 
We use analytic derivatives  all the time in real life.  Anyway, so however we calculate the  derivatives, we can now calculate a new slope. 
So our new slope will be equal to the previous  slope minus the derivative times the learning 
rate, which we've just set here at 0.0001. And we can do the same thing for the intercept  
as you see. And so here's our new slope intercept.  So we can use that for the second row of data. So the second row of data is x equals 86,  
y equals 202. So our intercept is not one, one anymore.  Intercept and slope are not one,  one, but they're 1.01 and 1.12. 
So here's, we're just using a formula just  to point at the new intercept and slope. 
We can get a new prediction and squared error  and derivatives, and then we can get another 
new slope and intercept. And so that was a pretty good one, actually. 
It really helped our slope head in the right  direction, although the intercept's moving  pretty slowly. And so we can do that for every row of data. 
Now strictly speaking, this is not mini  batch gradient descent that we normally  
do in deep learning.  It's a simpler version where  every batch is a size one.  So I mean, it's still stochastic gradient descent. It's just not, it's just a batch size of one. 
I think sometimes it's called online  gradient descent, if I remember correctly.  So we go through every data point in our very  small data set until we get to the very end. 
And so at the end of the first epoch, we've  got an intercept of 1.06 and a slope of 2.57. 
And those indeed are better estimates  than our starting estimates of 1, 1. 
So what I would do is I would copy our slope  2.57 up to here, 2.57, I'll just type it for 
now. And I'll copy our intercept up to here. 
And then it goes through the entire epoch  again, and we get another intercept and slope. 
And so we could keep copying and pasting  and copying and pasting again and again.  And we can watch the root  mean squared error going down. 
Now that's pretty boring doing  that copying and pasting.  So what we could do is  
fire up Visual Basic for applications. And sorry, this might be a bit small. 
I'm not sure how to increase the font size. And what it shows here. 
So sorry, this is a bit small. So you might want to just open it on   your own computer to be able to see it clearly. But basically it shows I've created a little  
macro where if you click on the reset button, it's just going to set the slope and constant to 1  
and calculate. And if you click the run button, it's   going to go through five times, calling one step. And what one step is going to do is it's going to  
copy the slope, last slope to the new slope and the last constant intercept  
to the new constant intercept. And also do the same for the RMSE. 
And it's actually going to paste it down to  the bottom for reasons I'll show you in a  moment. So if I now  
run this, reset and then run. There we go. 
You can see it's run at five times. And each time it's pasted the RMSE.  And here's a chart of it showing it going down. So you can see the new slope is 2.57. 
Your intercept is 1.27. I could keep running at another five.  So this is just doing copy, paste,  copy, paste, copy, paste; five times. 
And you can see that the RMSE is  very, very, very slowly going down. 
And the intercept and slope are very, very,  very slowly getting closer to where they want  to be. The big issue really  
is that the intercept is meant to be 30. It looks like it's going to take a very,   very long time to get there, but it will get there eventually if you click run enough times or maybe  
set the VBA macro to loop more than five steps at a time. 
But you can see it's very slowly. And importantly though, you can see like  
it's kind of taking this linear route every time these are increasing.  So why not increase it by more and more and more? And so you'll remember from last week  
that that is what momentum does. So on the next sheet, we show momentum. 
Momentum
And so everything's exactly the same as the  previous sheet, but this sheet, we didn't  bother with the finite differencing. We just have the analytic derivatives,  
which are exactly the same as last time. The data's the same as last time.  The slope and intercept are the  same starting points as last time. 
And this is the new b and new b that we get. 
But what we do this time is that we've added  a momentum term, which we're calling beta. 
And so the beta is going to these cells here. 
And what are these cells? Well, what these cells are is that they're,   maybe it's most interesting to take this one here. 
What it's doing is it's taking the gradient  and it's using that to update the weights, 
but it's also taking the previous update. So you can see here, the blue one, minus 25. 
So that is going to get  multiplied by 0.9, the momentum. 
And then the derivative is then multiplied by 0.1. 
So this is momentum, which is  getting a little bit of each.  And so then what we do is we then  use that instead of the derivative  
to multiply by our learning rate.  So we keep doing that again and  again and again, as per usual. 
And so we've got one column, which is calculating  the momentum, you know, lerped version of 
the gradient for both b and for a. And so  you can see that for this one, it's the same 
thing. It look at what was the previous move.  And that's going to be 0.9 of what you're going  to use for your momentum version gradient. 
And 0.1 is for this version,  the momentum gradient.  And so then that's again, what we're going  to use to multiply by the learning rate. 
And so you can see what happens is when you  keep moving in the same direction, which here 
is we're saying the derivative is  negative again and again and again.  So it gets higher and higher and higher. 
And ditto over here. And so particularly with this big jump   we get, we keep getting big jumps because still we want to, there's still negative gradient,  
negative gradient, negative gradient. So if we, at the end of this,  
our new, our b and our a have jumped ahead. And so we can click run. 
And we can keep clicking it. And you can see that it's moving, you know,  
not super fast, but certainly faster than it was before. 
So if you haven't used VBA, Visual Basic  for applications before, you can hit Alt,  
Alt F11 or Option F11 to open it.  And you may need to go into your preferences  and turn on the developer tools so that you 
can see it. You can also right click and choose assign  
macro on a button and you can see what macro has been assigned. 
So if I hit Alt F11 and I can just double,  or you can just double click on the sheet  name and it'll open it up. And you can see that this is  
exactly the same as the previous one. There's no difference here. 
Oh, one difference is that to keep  track of momentum at the very, very end. 
So I've got my momentum  values going all the way down.  The very last momentum I copy back up to the  top, each epoch so that we don't lose track 
of our kind of optimizer state, if you like. Okay.  So that's what momentum looks like. So, yeah, if you're kind of a more of a  
visual person like me, you like to see everything laid out in front of you and like to be able to   experiment, which I think is a good idea. This can be really helpful. 
RMSProp
So RMSProp, we've seen, and it's very similar  to momentum, but in this case, instead of 
keeping track of kind of a lerped moving average  and exponential moving average of gradients,  we're keeping track of a moving  average of gradients squared. 
And then rather than simply adding that, using  that as the gradient, what instead we're doing 
is we are dividing our gradient  by the square root of that. 
And so remember the reason we were doing that  is to say, if there's very little variation, 
very little going on in your gradients,  then you probably want to jump further. 
So that's RMSProp. And then finally,   Adam, remember, was a combination of both. 
Adam
So in Adam, we've got both the lerped version  of the gradient and we've got the lerped version 
of the gradient squared. And then we do both. 
When we update, we're both dividing  the gradient by the square root   of the lerped- moving, initially weighting average, moving averages. 
And we're also using the momentumized version. And so again, we just go through that each time. 
And so if I reset, run,  
and so, oh wow, look at that.  It jumped up there very quickly because  remember we wanted to get to 2 and 30. 
So just two sets. So that's 5, that's 10 epochs.  Now if I keep running it, it's  kind of now not getting closer. 
It's kind of jumping up and down  between pretty much the same values.  So probably what we'd need to do is  decrease the learning rate at that point. 
And yeah, that's pretty good. And now it's jumping up and down   between the same two values again. So it'd probably decrease the  
learning rate a little bit more. And I kind of like playing around like this   because it gives me a really intuitive feeling for what training looks like. 
So I've got a question from our YouTube  chat, which is how is J33 being initialized? 
So this is just, what happens is we take the  very last cell, well actually all these last 
four cells, and we copy them to here as values. So this is what those looked like  
in the last epoch. So basically we go copy and then  
paste as values. And then this here just refers back to them  
as you see.  And it's interesting that they're kind of,  you can see how they're exact opposites of 
each other, which is really,  you can really see how they're,   it's just fluctuating around the actual optimum at this point. 
Okay, thank you to Sam Watkins. We've now got a nicer sized editor. 
That's great. Where are we Adam? 
Okay, so with Adam, basically it all looks  pretty much the same, except now we have to 
copy and paste both our momentums and our  
squared gradients, and of course the slopes and intercepts at the end of each step.  But other than that, it's  just doing the same thing. 
And when we reset it, it just sets  everything back to their default values.  Now one thing that occurred to me, you know,  when I first wrote this spreadsheet a few 
years ago was that manually changing  the learning rate seems pretty annoying. 
Now of course we can use a scheduler, but  a scheduler is something we set up ahead of  time. And I did wonder if it's  
possible to create an automatic scheduler. And so I created this Adam annealing tab,  
Adam with annealing tab
which honestly I've never really got back to experimenting with.  So if anybody's interested,  they should check this out. 
What I did here was I used exactly the same  spreadsheet as the Adam spreadsheet, but I 
added an extra, after I do this step, I added an  extra thing, which is I automatically decreased 
the learning rate in a certain situation. And the situation in which I decreased it was I  
kept track of the average of the squared gradients. 
And anytime the average of the squared gradients  decreased during an epoch, I stored it. 
So I basically kept track of the  lowest squared gradients we had. 
And then what I did was if we got a… if that  resulted in the gradients, the squared gradients 
average halving, then I would  decrease the learning rate by,  
then I would decrease the learning rate by a factor of 4.  So I was keeping track of this gradient ratio. Now when you see a range like this, you can find  
what that's referring to by just clicking up here and finding gradient ratio. 
And there it is, and you can see that it's  equal to the ratio between the average of 
the squared gradients versus the  minimum that we've seen so far. 
So this is kind of like, my theory here was  thinking that, yeah, basically as you train, 
you kind of get into flatter  and more stable areas.  And as you do that, that's a sign that, you  know, you might want to decrease your learning 
rate. So yeah, if I try that, if I hit run  
again, it jumps straight to a pretty good value, but I'm not going to change the   learning rate manually. I just press run and you can see  
it's changed the learning rate automatically now. And if I keep hitting run without doing anything,  
look at that. It's got pretty good, hasn't it?  And the learning rates got lower and lower  and we basically got almost exactly the right 
answer. So yeah, that's a little experiment I tried.  So maybe some of you should try experiments  around whether you can create an automatic 
annealer using miniai. I think that would be fun. 
Learning Rate Annealing in PyTorch
So that is an excellent segue into our  notebook because we are going to talk   about annealing now. 
So we've seen it manually before  where we've just decreased the  
learning rate in a notebook and like run a second cell.  And we've seen something in Excel. But let's look at what we generally do in PyTorch. 
So we're still in the same notebook as  last time, the Accelerated SGD notebook. 
And now that we've re-implemented all the  main optimizers that people tend to use most 
of the time from scratch, we  can use PyTorch's, of course. 
So let's see, look now at how we can   do our own learning rate scheduling or annealing within the miniai framework. 
Now we've seen when we implemented the learning  rate finder that we saw how to create something 
that adjusts the learning rate. So just to remind you, this was all we had to do. 
So we had to go through the optimizers’ parameter  groups, and in each group set the learning 
rate to times equals some multiplier. That was for the learning rate finder. 
So since we know how to do that, we're not going  to bother re-implementing all the schedulers  from scratch because we know the basic idea now. So instead, what we're going to do is have a  
look inside the torch.optim.lr_scheduler module and see what's defined in there. 
So the lr_scheduler module, you can  hit dot, tab and see what's in there. 
But something that I quite like to do is to  use dir because dir(lr_scheduler) is a nice 
little function that tells you  everything inside a Python object. 
And this particular object is a module object  and it tells you all the stuff in the module. 
When you use the dot version tab, it doesn't  show you stuff that starts with an underscore, 
by the way, because that stuff's considered  private, or else dir does show you that stuff.  I can kind of see from here  that the things that start with  
a capital and then a small letter look like the things we care about.  We probably don't care about this. We probably don't care about these. 
So we can just do a little list comprehension  that checks that the first letter is an uppercase 
and the second letter is lowercase, and  then join those all together with a space.  And so here is a nice way to get a list of all  of the schedulers that PyTorch has available. 
And actually, I couldn't find such a list  on the PyTorch website in the documentation. 
So this is actually a handy  thing to have available. 
So here's various schedulers we can use. And so I thought we might experiment with using  
Cosine Annealing. 
So before we do, we have to recognize that  these PyTorch schedulers work with PyTorch 
optimizers, not with, of course,  with our custom SGD class.  And PyTorch optimizers have  a slightly different API. 
How PyTorch’s Optimizers work?
And so we might learn how they work. So to learn how they work, we need an optimizer. 
So one easy way to just grab an optimizer  would be to create a learner, just kind of 
pretty much any old random learner, and pass  in that single batch callback that we created. 
Do you remember that single batch callback? SingleBatch.  It just, after_batch, it cancels the fit. So it literally just does one batch. 
And we could fit. And from that,   we've now got a learner and an optimizer. And so we can do the same thing. 
We can do our optimizer to  see what attributes it has.  This is a nice way, or of course, just  read the documentation in PyTorch. 
This one is documented, I think,  showing all the things it can do.  As you would expect, it's got the step and  the zero_grad, like we're familiar with. 
Or you can just, if you just hit opt. So you can, the optimizers in PyTorch do  
actually have a repr, as it's called, which means you can just type it in and hit shift enter,   and you can also see the information about it this way. 
Now, an optimizer, it'll tell  you what kind of optimizer it is.  And so in this case, the default optimizer  for a learner, when we created it, we decided 
was optm.sgd.SGD. So we've got an SGD optimizer. 
And it's got these things called parameter groups. What are parameter groups?  Well, parameter groups are, as it  suggests, they're groups of parameters. 
And in fact, we only have one parameter group  here, which means all of our parameters are 
in this group. So let me kind of try and show you.  It's a little bit confusing,  but it's kind of quite neat. 
So let's grab all of our parameters. And that's actually a generator. 
So we have to turn that into an iterator and  call next, and that will just give us our  first parameter. Okay. 
Now what we can do is we can then  check the state of the optimizer. 
And the state is a dictionary and  the keys are parameter tensors. 
So this is kind of pretty interesting because  you might be, I'm sure you're familiar with  dictionaries. I hope you're familiar with dictionaries. 
But normally you probably use numbers or  strings as keys, but actually you can use  
tensors as keys.  And indeed that's what happens here. If we look at param, it's a tensor. 
It's actually a parameter, which remember  is a tensor, which it knows to requires_grad 
and to list in the parameters of the module. And so we're actually using that  
to index into the state. So if you look at opt.state,  
it's a dictionary where the keys are parameters. Now what's this for? 
Well, what we want to be able  to do is, if you think back to  
this, we actually had each parameter, we have state for it.  We have the average of the gradients or the  exponentially weighted moving average gradients 
and of squared averages. And we actually stored them as attributes. 
So PyTorch does it a bit differently. It doesn't store them as attributes, but instead  
the optimizer has a dictionary where you can index into it using a parameter  
and that gives you the state. And so you can see here, this  
is the exponentially weighted moving averages. And both because we haven't done any training  
yet and because we're using non-momentum SGD, it's none, but that's how it would be stored. 
So this is really important to  understand PyTorch optimizers.  I quite liked our way of doing it, of just  storing the state directly as attributes. 
But this works as well and it's fine. You just have to know it's there. 
And then, as I said, rather than  just having parameters, so we in SGD  
stored the parameters directly.  But in PyTorch, those parameters  can be put into groups. 
And so since we haven't put them into  groups, the length of param_groups is 1,  
this is one group.  So here is the param_groups. And that group contains all of our parameters. 
Okay. So pg, just to clarify here what's going on,  
pg is a dictionary. It's a parameter group. 
And to get the keys from a  dictionary, you can just listify it.  That gives you back the keys. And so this is one quick way  
of finding out all the keys in a dictionary. But you can see all the parameters in the group.  And you can see all of the hyper parameters,  the learning rate, the momentum, weight decay, 
and so forth.  So that gives you some background about  what's going on inside an optimizer. 
So Siva asks, isn't indexing by tensor just  like passing a tensor argument to a method? 
And no, it's not quite the  same, because this is state.  So this is how the optimizer  stores state about the parameters. 
It has to be stored somewhere. For our homemade miniai version,   we stored it as attributes on the parameter. In the PyTorch optimizers,  
they store it as a dictionary. So it's just how it's stored. 
Okay. So with that   in mind, let's look at how schedulers work. So let's create a cosine annealing scheduler. 
How schedulers work?
So a scheduler in PyTorch, you  have to pass at the optimizer.  And the reason for that is we want to be able  to tell it to change the learning rates of 
our optimizer. So it needs to know   what optimizer to change the learning rates of. So it can then do that for each set of parameters. 
And the reason that it does it by parameter  group is that as we'll learn in a later lesson, 
for things like transfer learning, we often  want to adjust the learning rates of the later  layers differently to the earlier layers  and actually have different learning rates. 
And so that's why we can have different groups  and the different groups have the different  learning rates, momentums, and so forth. Okay. 
So we pass in the optimizer and then, if I  hit shift tab a couple of times, it'll tell 
me all of the things that you can pass in. And so it needs to know T_max:  
how many iterations you're going to do. And that's because it's trying to do one,   you know, half a wave, if you like, of the cosine curve. 
So it needs to know how many  iterations you're going to do.  So it needs to know how far to step each time. So if we're going to do a hundred iterations. 
So the scheduler is going to  store the base learning rate.  And where did it get that from? It got it from our optimizer,  
which we set a learning rate. Okay.  So it's going to steal the optimizers learning  rate, and that's going to be the starting 
learning rate, the base learning rate. And it's a list because there could be a   different one for each parameter group. We only have one parameter group. 
You can also get the most recent learning  rate from a scheduler, which of course is 
the same. And so I couldn't find any method in PyTorch  
Plotting learning rates from a scheduler
to actually plot a scheduler's learning rates. So I just made a tiny little thing that just  
created a list, set it to the last learning rate of the scheduler, which is going to start  
at 0.006, and then goes through however many steps you ask for.  Steps the optimizer, steps the scheduler. So this is the thing that causes the scheduler  
to adjust its learning rate, and then just append that new learning rate to a list  
of learning rates and then plot it. So that's here's, and what I've done   here is I've intentionally gone over a hundred because I had told it I'm going to do a hundred. 
So I'm going over a hundred and you can see the  learning rate, if we did a hundred iterations,  would start high for a while. It would then go down and  
then it would stay low for a while. And if we intentionally go past the maximum,   it's actually start going up again because this is a cosine curve. 
So one of the main things, I guess, I wanted  to show here is like what it looks like to 
really investigate in a REPL  environment, like a notebook,  
how an object behaves, what's in it. 
And this is something I would always want  to do when I'm using something from an API 
I'm not very familiar with. I really want to like see what's in it,   see what they do, run it totally independently, plot anything I can plot. 
This is how I like to learn  about the stuff I'm working with. 
You know, data scientists don't spend all  of their time just coding, you know, so that 
means we need, we can't just rely on  using the same classes and APIs every day. 
So we have to be very good at  exploring them and learning about them.  And so that's why I think this  is a really good approach. 
Okay. So let's create a scheduler callback. 
Creating a scheduler callback
So a scheduler callback is something we're  going to pass in the scheduling class. 
But remember then when we… ah, the scheduling  callable, actually, and remember that when 
we create the scheduler, we have to  pass in the optimizer to schedule. 
And so before_fit, that's the point at which we  have an optimizer, we will create the scheduling 
object. I like this, “schedo”, it's very Australian.  So the scheduling object we will create by  passing the optimizer into the scheduler callable. 
And then when we do step(), then we'll check  if we're training and if so, we'll step(). 
Okay. So then what's   going to call step is after_batch. So after_batch, we'll call step. 
And that would be if you want your scheduler  to update the learning rate every batch. 
We could also have an epoch  scheduler callback, which we'll see 
later and that's just going to be after_epoch. Okay. 
So in order to actually see what the schedule  is doing, we're going to need to create a  new callback to keep track of  what's going on in our learner. 
And I figured we could create a recorder callback.  And what we're going to do is we're going  to be passing in the name of the thing that 
we want to record, that we want to keep track  of in each batch and a function, which is 
going to be responsible for  grabbing the thing that we want.  And so in this case, the function here is  going to grab from the callback, look up its 
param groups property and grab the learning rate. Where does the pg property come from? 
Or attribute? Well, before_fit the recorder callback is   going to grab just the first parameter group. Just so it's like, you've got to pick  
some parameter group to track. So we'll just grab the first one.  And so then also we're going to create a  dictionary of all the things that we're recording. 
So we'll get all the names. So that's going to be in this case, just lr.  And initially it's just going to be an empty list. And then after_batch, we'll go through each of the  
items in that dictionary, which in this case is just lr is the key   and underscore _lr function as the value. And we will append to that list, call that method,  
call that function or callable and pass in this callback.  And that's why this is going to get the callback. And so that's going to basically then have a  
whole bunch of, you know, dictionary of the results, you know, of each of these  
functions after each batch during training. So we'll just go through and plot them all. 
And so let me show you what  that's going to look like.  If we… let's create a cosine annealing callable. So we're going to have to use a partial to say  
Training with Cosine Annealing
that this callable is going to have T_max equal to three times, however many mini  
batches we have in our data loader. That's because we're going to do 3 epochs. 
And then we will set it running and  we're passing in the batch scheduler  
with the scheduler callable.  And we're also going to pass in our recorder  callback saying we want to track the learning 
rate using the _lr function. We're going to call fit.  And oh, this is actually a pretty good accuracy. We're getting, you know, close to 90%  
now in only 3 epochs, which is impressive. And so when we then call rec.plot, it's going to  
call, remember the rec is the recorder callback.  So it plots the learning rate. Isn't that sweet? 
So we could, as I said, we can do exactly  the same thing, but replace after_batch with  after_epoch. And this will now become a scheduler which  
steps at the end of each epoch rather than the end of each batch.  So I can do exactly the same thing  now using an epoch scheduler. 
So this time T_max is 3 because we're  only going to be stepping three times.  We're not stepping at the end of each  batch, just at the end of each epoch. 
So that trains and then we can  call rec.plot after trains. 
And as you can see there,  it's just stepping 3 times.  So you can see here, we're really digging  in deeply to understanding what's happening 
in everything in our models. What are all the activations look like?  What are the losses look like? What do our learning rates look like? 
And we've built all this from scratch. So yeah, hopefully that gives you a sense  
that we can really, yeah, do a lot ourselves. Now, if you've done the fastai Part 1 course,  
1-Cycle learning rate
you'll be very aware of 1-Cycle training, which was from a terrific paper by Leslie Smith,  
which I'm not sure it ever got published, actually. 
And 1-Cycle training is,  well, let's take a look at it. 
So we can just replace our  scheduler with OneCycleLR scheduler. 
So that's in PyTorch. And of course, if it wasn't in   PyTorch, we could very easily just write our own. I’m going to make it a batch scheduler and we're  
going to train, this time we're going to do 5 epochs.  So we're going to train a bit longer. And so the first thing I'll point out is, hooray,  
we have got a new record for us, 90.6%. So that's great. 
So, and then b), you can see here's the plot. And now look, two things are being plotted. 
And that's because I've now passed into the  recorder callback, a plot of learning rates  and also a plot of momentums. And momentums is going to grab the betas zero,  
because remember for Adam, it's called beta zero and beta one.  It's momentum of the gradients and  the momentum of the gradient squared. 
And you can see what the one cycle is doing  is the learning rate is starting very low 
and going up to high and then down again. But the momentum is starting high and then  
going down and then up again. So what's the theory here?  Well the starting out at a low learning rate  is particularly important if you have a not 
perfectly initialized model, which almost  everybody almost always does, even though 
we spent a lot of time  learning to initialize models.  We use a lot of models that get more complicated. And after a while people learn or  
figure out how to initialize more complex models properly.  So for example, this is a very, very cool paper. In 2019, this team figured out how  
to initialize ResNets properly. We'll be looking at ResNets very shortly.  And they discovered when they did  that they did not need batch norm. 
They could train networks of 10,000 layers. And they could get state of the  
art performance with no batch norm. And there's actually been something similar  
for transformers called T-Fixup that does a similar kind of thing. 
But anyway, it is quite difficult to initialize   models correctly. Most people fail to. 
Most people fail to realize that they generally  don't need tricks like warmup and batch norm 
if they do initialize them correctly. In fact, T-Fixup explicitly looks at this.  It looks at the difference between  no warmup versus with warmup  
with their correct initialization versus with normal initialization.  And you can see these pictures they're  showing are pretty similar actually. 
They're very similar to the  colorful dimension plots.  I kind of like our colorful dimension plots  better in some ways because I think they're 
easier to read, although I think  theirs are probably prettier.  So there you go, Stefano. There's something to inspire you if you want  
to try more things with our colorful dimension plots.  I think it's interesting that some papers  are actually starting to use a similar idea. 
I don't know if they got it from us  or they came up with it independently. 
But so we do a warmup if our networks not  quite initialized correctly, then starting 
at a very low learning rate means it's not  going to jump off way outside the area where 
the weights even make sense. And so then you gradually increase them as   the weights move into a part of the space that does make sense. 
And then during that time, while we have low  learning rates, if they keep moving in the 
same direction, then with this very high  momentum, they'll move more and more quickly.  But if they keep moving in different directions,  it's just the momentum is going to kind of 
look at the underlying direction they're moving. And then once you have got to a good part of the  
weight space, you can use a very high learning rate.  And with a very high learning rate,  you wouldn't want so much momentum. 
So that's why there's low momentum during  the time when there's high learning rate.  And then as we saw in our spreadsheet, which  did this automatically, as you get closer 
to the optimal, you generally want  to decrease the learning rate.  And since we're decreasing it  again, we can increase the momentum. 
So you can see that starting from random weights,  we've got a pretty good accuracy on Fashion 
MNIST with a totally standard convolutional  neural network, no ResNets, nothing else, 
everything built from scratch by hand, artisanal  neural network training, and we've got 90.6% 
for Fashion-MNIST. So there you go. 
All right, let's take a seven minute break. And I'll see you back shortly. 
I should warn you, you've got a lot more to cover. So I hope you're okay with a long lesson today. 
Okay, we're back. I just wanted to mention also something  
HasLearnCB - passing learn as parameter
we skipped over here, which is this HasLearn callback. 
This is more important for the people  doing the live course than the recordings.  If you're doing the recording,  you will have already seen this. 
But since I created Learner, actually, Piotr  Czapla, I don't know how to pronounce your 
surname, sorry, Piotr, pointed out that there's  actually kind of a nicer way of handling Learner… 
previously we were putting the Learner object  itself into self.learn in each callback. 
And that meant we were using self.learn.model  and self.learn.opt and self.learn dot all this 
all over the place. It was kind of ugly.  So we've modified Learner this week to instead  
pass in when it calls the callback,  
when in run_cbs, which is what it calls, Learner calls,   you might remember, is it passes the Learner as a parameter to the method. 
So now the Learner no longer goes through the  callbacks and sets their .learn attribute. 
But instead in your callbacks, you have to put  learn as a parameter in all of the callback 
methods. So for example,  
DeviceCB has a before_fit. So now it's got comma learn here. 
So now this is not self.learn. It's just learn. 
So it does make a lot of the code less yucky  to not have all this self.learn.pred equals 
self.learn.model, self.learn.batch. It's now just learn.  It also is good because you don't generally want  to have, both have the Learner has a reference 
to the callbacks, but also the  callbacks having a reference back   to the Learner, it creates something called a cycle. 
So there's a couple of benefits there. And that reminds me, there's a few   other little changes we've made to the code. And I want to show you a cool little trick. 
Changes from last week, /compare in GitHub
I want to show you a cool little trick for how  I'm going to find quickly all of the changes 
that we've made to the code in the last week. So to do that, we can go to the course repo. 
And on any repo, you can  add slash compare in GitHub.  And then you can compare across, you  know, all kinds of different things. 
But one of the examples they've got here  is to compare across different times.  Look at the master branch now versus one day ago. So I actually want the master  
branch now versus seven days ago. So I just hit this, change this to 7. 
And there we go. There's all my commits and I can immediately see   the changes from last week. And so you can basically see what are  
the things I had to do when I changed things. So for example, you can see here, all of my  
self.learn(s) became learn(s). I added the anneal, that's right, augmentation. 
And so in learner, I added an lr_find. Ah yes, I will show you that one.  That's pretty fun. So here's the changes we made to run_cbs, to fit. 
So this is a nice way I can quickly, yeah,  find out what I've changed since last time 
and make sure that I don't forget  to tell you folks about any of them.  Oh yes, cleanup_fit. I have to tell you about that as well. 
Okay. That's a useful reminder.  So the main other change to mention is that  calling the learning rate finder is now easier 
fastcore’s patch to the Learner with lr_find
because I added what's called  a patch to the learner.  fastcore’s patch decorator lets you take a  function and it will turn that function into 
a method of this class, of whatever  class you put after the colon. 
So this has created a new method  called lr_find or Learner.lr_find. 
And what it does is it calls self.fit, where  self is a Learner, passing in however many 
epochs you set as the maximum, check for your  learning rate finder, what to start the learning  rate at. And then it says to use as callbacks  
the learning rate finder callback. Now this is new as well. 
self.learn.fit didn't used to  have a callbacks parameter. 
So that's very convenient because what it  does is it adds those callbacks just during 
the fit. So if you pass in callbacks,   then it goes through each one  and appends it to self.cbs 
and when it's finished  fitting, it removes them again.  So these are callbacks that are just added  for the period of this one fit, which is what 
we want for learning rate finder. It should just be added for that one fit.  So with this patch in place, all that's required  to do the learning rate finder is now to create 
your learner and call.lr_find. And there you go.  Bang. So patch is a very convenient thing. 
It's one of these things which Python has a  lot of kind of like folk wisdom about what 
isn't considered Pythonic or good. And a lot of people really don't like patching. 
In other languages, it's used very  widely and is considered very good.  So I don't tend to have strong opinions  either way about what's good or what's bad. 
In fact, instead I just figure out  what's useful in a particular situation.  So in this situation, obviously it's very  nice to be able to add in this additional 
functionality to our class. So that's what.lr_find is. 
And then the only other thing we added to  the Learner this week was we added a few more 
New fit() parameters
parameters to fit. fit used to just take the number of epochs.  As well as the callback parameter, it  now also has a learning rate parameter. 
And so you've always been able to provide  a learning rate to the constructor, but you 
can override the learning rate for one fit. So if you pass in the learning rate,  
it will use it if you pass it in. And if you don't, it'll use the   learning rate passed into the constructor. And then I also added these two booleans to  
say when you fit, do you want to do the training loop and do you want to do the validation loop?  So by default, it'll do both. And you can see here, there's just  
an if train, do the training loop. If valid, do the validation loop. 
I'm not even going to talk about this, but if  you're interested in testing your understanding  of decorators, you might want to think  about why it is that I didn't have to  
say with torch.no_grad, but instead I called   torch.no_grad parenthesis function. That will be a very, if you can get to a  
point that you understand why that works and what it does, you'll be on your way to understanding   decorators better. Okay. 
So that is the end of Excel SGD. 
ResNets
ResNets. Okay,   so we are up to 90 point, what was it? Yeah, let's keep track of this. 
Oh yeah, 90.6% is what we're up to.  Okay. So to remind you the model,  
actually, so we're going to open 13_resnet now. 
And we're going to do the usual  import and setup initially. 
And the model that we've been using is  the same one we've been using for a while,  
which is that it's a convolution   and an activation and an optional batch norm. And in our models, we were using batch norm and  
applying our weight initialization, the kaiming weight initialization. 
And then we've got convs that take the  channels from 1 to 8 to 16 to 32 to 64,  
and each one stride-2.  And at the end, we then do a flatten. And so that ended up with a one by one. 
So that's been the model  we've been using for a while.  So the number of layers is 1, 2, 3, 4. So 4, 4 convolutional layers with a  
maximum of 64 channels in the last one. So can we beat 90.9, about 90 and a half,  
90.6, can we beat 90.6%? So before we do a ResNet, I thought, well,   let's just see if we can improve the architecture thoughtfully. 
So generally speaking, more depth and more  channels gives the neural net more opportunity 
to learn. And since we're pretty good at initializing   our neural nets and using batch norm, we should be able to handle deeper. 
So one thing we could do is we could,  let's just remind ourselves of the  
previous version so we can compare,   is we could have our, go up to 128 parameters. Now the way we'd do that is we could make our very  
first convolutional layer have a stride of 1.  So that would be one that goes from the one  input channel to eight output channels, or 
eight filters, if you like. So if we make it a stride of 1,   then that allows us to have one extra layer. And then that one extra layer could again double  
the number of channels and take us up to 128.  So that would make it deeper and  effectively wider as a result. 
So we can do our normal BatchNorm2d and our  new OneCycle learning rate with our scheduler. 
And the callbacks we're going to use is the  Device callback, our Metrics, our Progress  bar, and our activation stats  looking for GeneralRelus. 
And I won't have you watch them train  because that would be kind of boring.  But if I do this with this deeper and eventually  wider network, this is pretty amazing. 
We get up to 91.7%. So that's like quite a big difference.  And literally the only difference to our previous  model is this one line of code, which allowed 
us to take this instead of going  from 1 to 64, it goes from 8 to 128. 
So that's a very small change,  but it massively improved.  So the error rate's gone down by  about over 10% relatively speaking  
in terms of the error rate.  So there's a huge impact we've already had. Again, 5 epochs. 
So now what we're going to do is  we're going to make it deeper still.  But it gets, there becomes a point. 
So Kaiming He et al. noted that there comes  a point where making neural nets deeper stops 
working well. And remember, this is the guy who created   the initializer that we know and love. And he pointed out that  
even with good initialization, there  comes a time where adding more layers 
becomes problematic. And he pointed out   something particularly interesting. He said, let's take a 20 layer neural network. 
This is in a paper called “Deep Residual Learning  for Image Recognition” that introduced ResNets. 
Let's take a 20 layer network and train it  for a few, what's that, tens of thousands 
of iterations and track its test error. Okay.  And now let's do exactly the same thing on  a 56 layer, otherwise identical, but deeper 
And he pointed out that the 56 layer  network had a worse error than the 20 layer. 
And it wasn't just a problem of generalization  because it was worse on the training   set as well. 
Now the insight that he had is if you just  set the additional 36 layers to just identity, 
you know, identity matrices, they  should, they would do nothing at all.  And so a 56 layer network is a  superset of a 20 layer network. 
So it should be at least as  good, but it's not, it's worse.  So clearly the problem here is  something about training it. 
And so him and his team came up with a really  clever insight, which is, can we create a 56 
layer network, which has the same training  dynamics as a 20 layer network or even less? 
And they realized, yes, you can. What you could do is you could add  
something called a shortcut connection. And basically the idea is that, normally,  
when we have, you know, our inputs coming into our convolution. 
So let's say that's, that was our inputs and  here's our convolution and here's our outputs. 
Now if we do this 56 times, that's a lot of  stacked up convolutions, which are effectively 
matrix multiplications with a  lot of opportunity for, you know,   gradient explosions and all that fun stuff. 
So how could we make it so  that we have convolutions, but  
with the training dynamics of a much shallower  network. And  
here's what he did. He said, let's actually put two convs in here  
to make it twice as deep because we are trying to make things deeper, but then let's add  
what's called a skip connection where instead of just being out equals, so this is conv1,  
this is conv2. Instead of being out equals and there's a, you know, assume that  
these include activation functions, equals conv2 of conv1 of in, right? 
Instead of just doing that, let's  make it conv2 of conv1 of in plus  
in. Now if we initialize  
these at the first to have weights  of zero, then initially this 
will do nothing at all. It will output zero and therefore at first  
you'll just get out equals in, which is exactly what we wanted, right? 
We actually want to, for it to be  as if there is no extra layers. 
And so this way we actually end up with a  network which can be deep, but also at least 
when you start training  behaves as if it's shallow.  It's called a residual connection because  if we subtract in from both sides,  
then we would get out minus in   equals conv1 of conv2 of in. In other words, the difference between the  
endpoint and the starting point, which is the residual.  And so another way of thinking about it  is that this is calculating a residual. 
So there's a couple of ways of thinking about it. And so this thing here  
is called the Res Block or ResNet Block. 
Okay, so Sam Watkins has just pointed out  the confusion here, which is that this only 
works if, let's put the minus in  back and put it back over here. 
This only works if you can add these together. Now if conv1 and conv2 both have the same number  
of channels as in the same number of filters, same number of filters,  
and they also have stride-1, then that will work fine. 
You'll end up, that will be exactly the same  output shape as the input shape and you can  add them together. But if they are  
not the same, then you're in a bit of trouble. So what do you do? 
And the answer which Kaiming He et al. came  up with is to add a conv on in as well,  
but to make it as simple as possible. 
We call this the identity conv. It's not really an identity anymore, but we're   trying to make it as simple as possible so that we do as little to mess up these  
training dynamics as we can. And the simplest possible convolution   is a 1 by 1 filter block, a 1 by 1 kernel, I guess we should call it. 
A 1 by 1 kernel size.  And using that, and we can also add  a stride or whatever if we want to. 
So let me show you the code. So we're going to create  
something called a conv block. Okay, and the conv block is going to do   the two convs. That's going to be a conv block. 
Okay, so we've got some number of input filters,  some number of output filters, some stride, 
some activation functions, possibly a  normalization and some kernel shape,  
some kernel size. Okay, so the second conv  
is actually going to go from  output filters to output filters  because the first conv is going to be  from input filters to output filters. 
So by the time we get to the second  conv, it's going to be nf to nf. 
The first conv, we will set stride-1, and  then the second conv will have the requested 
stride. And that way the two convs back to back   are going to overall have the requested stride. So this way, the combination of these two convs  
is going to eventually take us from ni to nf in terms of the number of filters,   and it's going to have the stride that we requested. 
So it's going to be a, the conv block is a  sequential block consisting of a convolution 
followed by another convolution, each  one with the requested kernel size. 
And the requested activation function  and the requested normalization layer.  The second conv won't have an activation function. I'll explain why in a moment. 
And so I mentioned that one way to make this  as if it didn't exist would be to set the 
convolutional weights to  zero and the biases to zero.  But actually we would like to have, you  know, correctly randomly initialized weights. 
So instead what we can do is  if you're using batch norm,  
we can initialize this conv2[1] will be the batch norm layer.  We can initialize the batch norm weights to zero. Now if you've forgotten what that means,  
go back and have a look at our implementation from scratch of batch norm because the batch norm   weights is the thing we multiply by. So do you remember the batch norm? 
We subtract the exponential moving average  mean, we divide by the exponential moving 
average standard deviation, but then we add  back the, the kind of the, the, the, the batch 
norms bias layer and we multiply by the batch  norms weights, well, other way around, multiply 
by weights first. So if we set the batch norm layers   weights to zero, we're multiplying by zero. And so this will cause the initial  
conv block output to be just all zeros. And so that's going to give us, what we wanted  
is that nothing's happening here. So we just end up with the input   with this possible idconv. So a ResBlock is going to  
contain those convolutions in the  convolution block we just discussed,  right? And then we're going to need this idconv. 
So the idconv is going to be a noop. So that's nothing at all. 
If the number of channels in is equal to the  number of channels out, but otherwise we're 
going to use a convolution with a  kernel size of 1 and a stride of 1.  And so that is going to, you know, is with  as little work as possible change the number 
of filters so that they match. Also what if the stride's not 1? 
Well if the stride is 2, actually this isn't  going to work for any stride, this only works  for a stride of 2. If there's a stride of 2,  
we will simply average using average pooling. So this is just saying take the mean  
of every set of 2 items in the grid. So we'll just take the mean. 
So we basically have here pool of idconv of  in if the stride is 2 and if the filtered 
number is changed. And so that's the minimal amount of work.  So here it is, here is the forward pass. We get our input and on the identity connection  
we call pool and if stride is 1, that's a noop, so do nothing at all. 
We do idconv and if the number of filters  is not changed, that's also a noop. 
So this is just the input in that situation. And then we add that to the result of the convs. 
And here's something interesting, we then  apply the activation function to the whole 
thing. Okay, so that way, I wouldn't say   this is like the only way you can do it, but this is a way that works pretty well, is to apply the  
activation function to the result of the whole resnet block. 
And that's why I didn't add activation  function to the second conv.  So that's a res block. So it's not a huge amount of code, right? 
And so now I've literally copied and pasted  our get_model, but everywhere that previously 
we had a conv, I've just  replaced it with ResBlock.  In fact, let's have a look. 
get_model. 
Okay, so previously we started with  conv 1 to 8, now we do ResBlock 1 to 8, 
stride-1, stride-1, then we added conv from  number of filters i to number of filters  i + 1, now it's ResBlock from number  of filters, number of filters i + 1. 
Okay, so it's exactly the same. One change I have made though, is I mean,  
it doesn't actually make any difference at all, I think it's mathematically identical,   is previously the very last conv at the end went from the 128 channels down to the 10  
channels, followed by flatten, but this conv is actually working on a 1 by 1 input. 
So in an alternate way, but I think makes  it clearer is flatten first and then use a 
linear layer because a conv on a 1 by  1 input is identical to a linear layer. 
And if that doesn't immediately make sense,  that's totally fine, but this is one of those  places where you should pause and have a little  stop and think about why a conv on a 1 by 
scratch conv we did, because  this is a very important insight.  So I think it's very useful with a more complex  model like this to take a good old look at 
it to see exactly what the inputs  and outputs of each layer is.  So here's a little function called _print_shape,  which takes the things that a hook takes and 
we will print out for each layer, the name  of the class, the shape of the input and the 
shape of the output. So we can get our model,   create our learner and use a  handy little Hooks context manager 
we built in an earlier lesson and  call the _print_shape function.  And then we will call fit for 1 epoch just  doing the evaluation not the training. 
And if we use the SingleBatch callback, it'll  just do a single batch, pass it through and 
that hook will, as you see, print out each  layer, the inputs shape and the output shape. 
So you can see we're starting with an input  of a batch size of 1024, 1 channel, 28 by 
Our first ResBlock was stride-1,  so we still end up with 28 by 28,   but now we've got 8 channels. 
And then we gradually decrease the grid  size to 14, to 7, to 4, to 2, to 1, 
as we gradually increase the number of channels. We then flatten it, which gets rid of that 1 by 1,  
which allows us then to do linear to counter the 10. 
And then there's some discussion about whether  you want a batch norm at the end or not. 
I was finding it quite useful in this case. So we've got a batch norm at the end. 
I think this is very useful. So I decided to create a patch  
for Learner called summary  that would do basically exactly  the same thing, but it would  do it as a markdown table. 
Training the ResNet
Okay. So, if we create a TrainLearner with our model  
and call dot summary, this method is now available 
because it's been patched  that method into the Learner. 
And it's going to do exactly the same thing  as our print, but it does it more prettily  by using a markdown table, if it's in a  notebook, otherwise it'll just print it. 
So fastcore has a handy thing for  keeping track if you're in a notebook.  And in a notebook to make something markdown,  you can just use IPython.display.Markdown 
as you see. And the other thing that I added as well   as the input and the output is, I thought, let's also add in the number of parameters. 
So we can calculate that as we've seen  before by summing up the number of elements  
for each parameter in that module.  And so then I've kind of kept track of that  as well so that at the end I can also print 
out the total number of parameters. So we've got a 1.2 million parameter model,  
and you can see that there's very few parameters here in the input. 
Nearly all the parameters are  actually in the last layer.  Why is that? Well, you might want to go back to  
our Excel convolutional spreadsheet to see this. You have a parameter, for every input channel,  
you have a set of parameters.  They're all going to get added up  across each of the 3 by 3 in the kernel. 
And then that's going to be done for  every output filter, every output channel  
that you want.  So that's why you're going to end  up with, in fact, let's take a look. 
Maybe let's create, let's  just grab some particular   one. So create our model. 
And so we'll just have a look at  
the sizes. And so you can see here  
there is this 256 by 256 by 3 by 3. So that's a lot of parameters. 
Okay. So we can call lr_find   on that and get a sense of what  kind of learning rate to use. 
So I chose 2e-2, so 0.02. This is our standard learning thing. 
You don't have to watch it train. I've just trained it.  And so look at this, by using  ResNet, we've gone up from 91.7 
This is just keeps getting better. So that's pretty nice. 
And you know, this ResNet is not anything fancy. It's the simplest possible ResBlock, right? 
The model is literally copied and pasted from  before and replace each place it said conv 
with ResBlock. But we've just been thoughtful about it,   you know, and here's something very interesting. We can actually try lots of other ResNets by  
ResNets from timm
grabbing timm. So that's Ross   Wightman's PyTorch image model library. And if you call timm.list_models(*resnet*),  
there's a lot of ResNets. And I tried quite a few of them. 
Now one thing that's interesting is, if you  actually look at the source code for timm, 
you'll see that the various different ResNets  like ResNet18, ResNet18d, ResNet10d, they're 
defined in a very nice way using  this very elegant configuration. 
You can see exactly what's different. So there's basically only one line of code   different between each different type of ResNet for the main ResNets. 
And so what I did was I tried all the timm  models I could find, and I even tried importing 
the underlying things and building  my own ResNets from those pieces. 
And the best I found was the ResNet18d. 
And if I train it in exactly  the same way, I got to 92%. 
And so the interesting thing is  you'll see that's less than our 92.2  And it's not like I tried  lots of things to get here. 
This was the very first thing I tried. Whereas this ResNet18d was after trying   lots and lots of different timm models. And so what this shows is that the just  
thoughtfully designed kind  of basic architecture goes a 
very long way. It's actually better for this problem than  
any of the PyTorch image models, ResNets, that I could find. 
So I think that's quite amazing, actually. It's really cool.  And it shows that you can create  a state-of-the-art architecture  
just by using some common sense. So I hope that's encouraging. 
So anyway, so we're up to 92.2%. We're not done yet. 
Because we haven't even talked  about data augmentation.  All right. So, let's keep going. 
Going wider
So we're going to make  everything the same as before.  But before we do data augmentation, we're  going to try to improve our model even further, 
if we can.  So I said it was kind of not constructed  with any great care and thought, really. 
Like in terms of this ResNet, we just took  the ConvNet and replaced it with a ResNet. 
So it's effectively twice as deep, because  each conv block has 2 convolutions. 
But ResNets train better than ConvNets. So surely we could go deeper and wider still. 
So I thought, OK, how could we go wider? And I thought, well, let's take our model. 
And previously, we were going from 8 up to 256. What if we could get up to 512? 
And I thought, OK, well, one way to do that  would be to make our very first ResBlock 
not have a kernel size of  3, but a kernel size of 5.  So that means that each  grid is going to be 5 by 5. 
That's going to be 25 inputs. So I think it's fair enough,   then, to have 16 outputs. So if I use a kernel size of 5, 16 outputs,  
then that means if I keep doubling as before, I'm going to end up at 512 rather than 256. 
OK, so that's the only change I made, was  to add ks equals 5 here and then change to 
double all the sizes. And so if I train that,  
wow, look at this, 92.7% So we're getting better still. 
And again, it wasn't with lots of  trying and failing and whatever.  It was just like saying,  well, this just makes sense. 
And the first thing I tried, it just worked. We're just trying to use  
these sensible, thoughtful approaches. Next thing I'm going to try isn't necessarily  
Pooling
something to make it better, but it's something to make our ResNet more flexible.  Our current ResNet is a bit  awkward in that the number of  
stride-2 layers has to be exactly big enough that the last of them  
ends up with a 1 by 1 output. So you can flatten it and do the linear. 
So that's not very flexible, because what  if you've got something of a different size? 
So to make that necessary, I've created  a get_model_2, which goes less far. 
It has one less layer. So it only goes up to 256, despite starting at 16.  And so because it's got one less layer, that  means that it's going to end up at the 2 by 
So what do we do? Well, we can do something very straightforward,   which is we can take the mean over the 2 by And so if we take the mean over the 2 by 2,  
that's going to give us… a mean over the 2 by 2,  it's going to give us batch size  by channels output, which is what  
we can then put into our linear layer.  So this is called, this ridiculously simple  thing, is called a Global Average Pooling 
layer. That's the Keras term.  In PyTorch, it's basically the same. It's called an Adaptive Average Pooling layer. 
But in PyTorch, you can cause it to  have an output other than 1 by 1. 
But nobody ever really uses it that way. So they're basically the same thing. 
This is actually a little bit more convenient  than the PyTorch version, because you don't  have to flatten it. So this is Global Average Pooling. 
So you can see here, after our  last ResBlock, which gives us   a 2 by 2 output, we have GlobalAvgPool And that's just going to take the mean. 
And then we can do the Linear, BatchNorm as usual.  So I wanted to improve my summary patch to  include not only the number of parameters, 
but also the approximate number of megaFLOPs. So FLOP is a floating operation per second,  
a floating point operation per second. I'm not going to promise my   calculation is exactly right. I think the basic idea is right. 
I just basically actually calculated. It's not really FLOPs.  I actually calculated the  number of multiplications. 
So this is not perfectly accurate,  but it's pretty indicative, I think.  So this is the same summary I had before,  but I added an extra thing, which is a _flops 
function, where you pass in the weight matrix  and the height and the width of your grid. 
Now if the number of dimensions of the weight  matrix is less than 3, then we're just doing 
like a linear layer or something. So actually just the number of elements is the   number of FLOPs, because it's just a matrix multiply. 
But if you're doing a convolution, so the  dimension is 4, then you actually do that 
matrix multiply for everything  in the height by width grid.  So that's how I calculate this  kind of FLOPs equivalent number. 
So okay, so if I run that on this model, we  can now see our number of parameters compared 
to the ResNet model has gone from  1.2 million up to 4.9 million. 
And the reason why is because we've got this,  we've got this ResBlock that gets all the 
way up to 512. And the way we   did this is we made that a stride-1 layer. So that's why you can see here it's gone 2,  
2 and it stayed at 2, 2. So I wanted to make it   as similar as possible to the last ones. It's got the same 512 final number of channels. 
And so most of the parameters are in that  last block for the reason we just discussed. 
Interestingly, though, it's  not as clear for the megaFLOPs. 
It is the greatest of them, but in terms of  number of parameters, I think this has more 
parameters than all the other  ones added together by a lot.  But that's not true of megaFLOPs. And that's because this first layer  
has to be done 28 by 28 times, whereas this layer only has to be done 2 by 2 times. 
Anyway, so I tried training that  and got pretty similar result, 92.6 
Reducing the number of parameters and megaFLOPS
And that kind of made me think, oh, let's  fiddle around with this a little bit more 
to see like what kind of things would reduce  the number of parameters and the megaFLOPs. 
The reason you care about reducing the number  of parameters is that it has a lower memory  requirements. 
And the reason you want to reduce the  number of FLOPs is it's less compute. 
So in this case, what I've done here  
is I've removed this line of code. So I've removed the line  
of code that takes it up to 512. So that means we don't have this layer anymore.  And so the number of parameters has gone  down from 4.9 million down to 1.2 million. 
Not a huge impact on the megaFLOPs,  but a huge impact on the parameters.  We've reduced it by like 2  thirds or 3 quarters or something  
by getting rid of that. And you can see that the,  
if we take the very first ResNet  block, the number of parameters 
is, you know, why is it this 5.3 megaFLOPs? Because although the very first one starts with  
just one channel, the first conv, remember our ResNet blocks have two convs.  So the second conv is going  to be a 16 by 16 by 5 by 5. 
And again, I'm partly doing this to show you  the actual details of this architecture, but  I'm partly showing it so that you can see  how to investigate exactly what's going on 
in your models. And I really want you to try these.  So if we train that one,  interestingly, even though  
it's only a quarter or something of the size, we get the same accuracy, 92.7 
So that's interesting. Can we make it faster? 
Well, at this point, this is the obvious place  to look at is this first ResNet block, because 
that's where all the megaFLOPs are. And as I said, the reason is   because it's got two convs. The second one is 16 by 16  
channels, 16 channels in, 16  channels out, and it's doing these 
And it's having to do it  across the whole 28 by 28 grid.  So that's the bulk of the biggest compute. 
So what we could do is we could replace  this ResBlock with just one convolution. 
And if we do that, then you'll  see that we've now got rid of  
the 16 by 16 by 5 by 5. We just got the 16 by 1 by 5 by 5.  So the number of megaFLOPs has  gone down from 18.3 to 13.3 
The number of parameters hasn't  really changed at all, right?  Because the number of parameters  was only 6,800, right? 
So be very careful that when you see people  talk about, oh, my model has less parameters. 
That doesn't mean it's faster. Okay.  Really, it doesn't mean that at all. There's no particular relationship  
between parameters and speed. Even counting megaFLOPs doesn't always   work that well, because it doesn't take account of the amount of things moving through memory. 
But it's not a bad approximation here. So here's one which has got much less megaFLOPs. 
And in this case, it's about  the same accuracy as well.  So I think this is really interesting. We've managed to build a model that has far  
less parameters and far less megaFLOPs and has basically exactly the same accuracy. 
So I think that's a really  important thing to keep in mind.  And remember, this is still way  better than the ResNet18d from timm. 
So we built something that  is fast, small, and accurate. 
Training for longer
So the obvious question is,  what if we train for longer?  And the answer is, if we train for longer,  if we train for 20 epochs, I'm not going to 
wait for it. The training accuracy gets up to 0.999 
But the validation accuracy is worse. It's 0.924 
And the reason for that is that after 20 epochs,  it's seen the same picture so many times, 
it's just memorizing them.  And so once you start memorizing,  things actually go downhill. 
So we need to regularize. Now something that we have claimed in   the past can regularize is to use weight decay. But here's where I'm going to point out that  
weight decay doesn't regularize at all if you use BatchNorm. 
And it's fascinating. For years, people didn't even seem to notice this.  And then somebody, I think, finally  wrote a paper that pointed this out. 
And people were like, oh, wow, that's weird. But it's really obvious when you think about it. 
A BatchNorm layer has a single set of  coefficients which multiplies an entire layer. 
So that set of coefficients could  just be the number 100 in every place. 
And that's going to multiply the entire previous  weight matrix or convolution kernel matrix 
by 100. As far as weight decay is concerned, that's  
not much of an impact at all because the BatchNorm layer has very few weights. 
So it doesn't really have a  huge impact on weight decay.  But it massively increases the  effective scale of the weight matrix. 
So BatchNorm basically lets the neural net  cheat by increasing the coefficients, the 
parameters, even nearly as much as it wants  indirectly just by changing the BatchNorm  layer's weights. So weight decay is not going to save us. 
And that's something really  important to recognize.  Weight decay is not, I mean, with BatchNorm  layers, I don't see the point of it at all. 
It does have some, like, there has  been some studies of what it does.  And it does have some weird kind of  second order effects on the learning rate. 
But I don't think you should rely on them. You should use a scheduler for changing the   learning rate rather than weird second order effects caused by weight decay. 
Data Augmentation
So instead, we're going to do data augmentation,  which is where we're going to modify every 
image a little bit by random change so that  it doesn't see the same image each time. 
So there's not any particular reason to  implement these from scratch, to be honest. 
We have implemented them  all from scratch in fastai.  So you can certainly look  them up if you're interested. 
But it's actually a little bit separate  to what we're meant to be learning about.  So I'm not going to go through it. 
But yeah, if you're interested,  go into fastai, vision, augment. 
And you'll be able to see, for  example, how do we do flip? 
And you know, it's just like x.transpose. Okay, which is not really,  
yeah, it's not that interesting. Yeah, how do we do cropping and padding? 
How do we do random crops, so on and so forth? Okay, so we're just going to actually, you know,   fastai has probably got the best implementation 
of these, but torchvision’s are fine. So we'll just use them. 
And so we've created before  a batch transform callback. 
And we used it for normalization, if you remember. So what we could do  
is we could create a transform batch function,  
which transforms the inputs and transforms the outputs  
using two different functions. So that would be an augmentation callback. 
And so then you would say, okay, for the transform  batch function, for example, in this case,  we want to transform our x's. And how do we want to transform our x's? 
And the answer is, we want to transform them  using this module, which is a sequential module 
of first of all doing a RandomCrop,  and then a RandomHorizontalFlip. 
Now it seems weird to randomly crop a 28 by  28 image to get a 28 by 28 image, but we can 
add padding to it. And so effectively, it's going to randomly   add padding on one or both sides to do this kind of random crop. 
One thing I did to change  the BatchTransform callback,  
can't remember if I've mentioned this before, but something I changed slightly   since we first wrote it, is I added this on_train and on_val so that it only does it if you said I  
want to do it on training and it's training, or I want to do it on   validation and it's not training. And then this is all the code is. 
So data augmentation, generally speaking,  shouldn't be done on validation, so we set 
on_val false. Okay, so  
what I'm going to do first of all is I'm going to use our classic SingleBatchCB trick   and fit, in fact, even better, oh yeah, fit(1) just doing training. 
And what I'm going to do then is after I  fit, I can grab the batch out of the learner. 
And this is a way, this is quite cool, right? This is a way that I can see exactly   what the model sees, right? So this is not relying on any,  
you know, approximations. Remember when we fit, it puts   it in the batch that it looks at into learn.batch. So if we fit for a single batch, we can then grab  
that batch back out of it and we can call show_images.  And so here you can see  this little crop it's added. 
Now something you'll notice is that every  single image in this batch, notice I grabbed  the first 16, so I don't want to show you  1024, has exactly the same augmentation. 
And that makes sense, right, because  we're applying a BatchTransform. 
Now why is this good and why is it bad? It's good because this is running  
on the GPU, right? Which is great because nowadays very often  
it's really hard to get enough CPU to feed your fast GPU fast enough. 
Particularly if you use something  like Kaggle or Colab that are   really underpowered for CPU, particularly Kaggle. 
So this way all of our transformations, all  of our augmentation is happening on the GPU. 
On the downside, it means that  there's a little bit less variety.  Every mini batch has the same augmentation. I don't think the downside matters though,  
because it's going to see lots of mini batches. So the fact that each mini batch is going to have   a different augmentation is actually all I care about. 
So we can see that if we run this multiple times,  
you can see it's got a different augmentation in each mini batch. 
Okay, so I decided actually I'm  just going to use 1 padding. 
So I'm just going to do a very, very  small amount of data augmentation.  And I'm going to do 20 epochs  using OneCycle learning rate. 
And so this takes quite a while to train,  so we won't watch it, but check this out. 
We get to 93.8 That's pretty wild. 
Yeah that's pretty wild. So I actually went on Twitter and I said  
to the entire world on Twitter, you know, which if you're watching this in 2023, if Twitter  
doesn't exist yet, ask somebody to tell you about what Twitter used to be.  Hopefully it still does. Can anybody beat this in 20 epochs? 
You can use any model you like, any library  you like, and nobody's got anywhere close. 
So this is pretty amazing. And actually, you know, when I had  
a look at papers with code, there are, you know, well, I mean, you can see it's right up there,  
right, with the kind of best models that are listed, certainly better than these ones. 
And the better models all use,  you know, 250 or more epochs. 
So yeah, if anybody, I'm hoping that somebody  watching this will find a way to beat this 
in 20 epochs, that would be really great. Because as you can see, we haven't really done   anything very amazingly weirdly clever. It's all very, very basic. 
And actually we can go even  a bit further than 93.8. 
Just before we do, I mentioned that since  this is actually taking a while to train now,  I can't remember, it takes like  10 to 15 seconds per epoch. 
So you know, you're waiting a few  minutes, you may as well save it.  So you can just call torch.save on a model,  and then you can load that back later. 
So something that can make things even better  is something called test time augmentation. 
Test Time Augmentation
I guess I should write this out properly here. Test, text, test time augmentation. 
Now test time augmentation actually does  our BatchTransform callback on validation as 
well. And then what we're going to do is we're actually,   in this case, we're going to do just a very, very, very simple test time augmentation,  
which is we're going to add a BatchTransform callback that runs on validate and it's not  
random, but it actually just does a horizontal flip.  Non-random, so it always does a horizontal flip. And so check this out. 
What we're going to do is we're going to  create a new callback called CapturePreds. 
And after each batch, it's  just going to append to a list  
the predictions, and it's going to append to a different list the targets. 
And that way we can just call learn.fit, train  equals False, and it will show us the accuracy. 
And this is just the same  number that we saw before.  But then what we can do is we can call the  same thing, but this time with a different 
callback, which is with the  horizontal flip callback. 
And that way it's going to do exactly the  same thing as before, but in every time it's  going to do a horizontal flip. And weirdly enough, that accuracy is slightly  
higher, which that's not the interesting bit. The interesting bit is that we've now got  
two sets of predictions.  We've got the sets of predictions  with the non-flipped version.  We've got the set of predictions  with the flipped version. 
And what we could do is we could stack  those together and take the mean. 
So we're going to take the average of  the flipped and unflipped predictions. 
And that gives us a better result still, 94.2% So why is it better? 
It's because looking at the image from kind  of like multiple different directions gives 
it more opportunities to try to  understand what this is a picture of.  And so in this case, I'm just giving it two  different directions, which is the flipped 
and unflipped version, and  then just taking their average. 
So yeah, this is like a really nice little trick. Sam's pointed out it's a bit  
like random forest, which is true. It's a kind of bagging that we're doing.  We're kind of getting multiple  predictions and bringing them together. 
And so we can actually,   so 94.2 I think is my best 20 epoch result. And notice I didn't have to  
do any additional training. So it still counts as a 20 epoch result.  You can do test time augmentation where you  do, you know, a much wider range of different 
augmentations that you trained with, and  then you can use them at test time as well.  You know, more, more crops or  rotations or warps or whatever. 
Random Erasing
I want to show you one of my favorite  data augmentation approaches, which is   called random erasing. 
So random erasing, I'll show you  what it's going to look like.  Random erasing, we're going to add a little,  we're going to basically delete a little bit 
of each picture and we're going to replace  it with some random Gaussian noise. 
Now, in this case, we've just got one patch. But eventually we're going   to do more than one patch. So I wanted to implement this because remember  
we have to implement everything from scratch. And this one's a bit less trivial than the  
previous transforms. So we should do it from scratch.  And also I'm not sure there's  that many good implementations. 
Ross Wightman's Tim I think has one. And so, and it's also a very   good exercise to see how to  implement this from scratch. 
So let's grab a batch out of the training set. And let's just grab the first 16 images. 
And so then let's grab the  mean and standard deviation.  Okay.  And so what we want to do is we wanted to  delete a patch from each image, but rather 
than deleting it, deleting it  would change the statistics, right?  If we set those all to zero, the mean and  standard deviation are now not going to be 
But if we replace them with exactly the same  mean and standard deviation pixels that the 
picture has, or that our dataset has,  then it won't change the statistics.  So that's why we've grabbed the  mean and standard deviation. 
And so we could then try grabbing,  let's say we want to delete 0.2,  
so 20% of the height and width.  Then let's find out how big that size is. So 0.2 of the shape, of the height and of  
the width, that's the size of the x and y. And then the starting point, we're just going  
to randomly grab some starting point, right? So in this case, we've got the starting point  
for x is 14, starting point for y is 0, and then it's going to be a 5 by 5 spot. 
And then we're going to do a Gaussian or  normal initialization of our mini batch,  
everything in the batch, every channel   for this x slice, this y slice,  and we're going to initialize 
it with this mean and standard  deviation, normal random noise. 
And so that's what this is. So it's just that tiny little bit of code. 
So you'll see, I don't  start by writing a function.  I start by writing single lines of code that  I can run independently and make sure that 
they all work and that I look at the  pictures and make sure it's working.  Now one thing that's wrong here is that you  see how the different, you know, this looks 
black and this looks gray. Now at first this was confusing   me as to what's going on. What's it changed? 
Because the original images didn't look like that. And I realized the problem is that the minimum  
and the maximum have changed. It used to be from -0.8 to 2.  That was the previous min and max. Now it goes from -3 to 3. 
So the noise we've added has the same mean  and standard deviation, but it doesn't have 
the same range because the pixels were  not normally distributed originally.  So normally distributed noise actually is wrong. So to fix that, I created a new version  
and I'm putting in a function now. It does all the same stuff as before,  
as I just did before, but it clamps the random pixels to be between min and max. 
And so it's going to be exactly the same thing,  but it's going to make sure that it doesn't  change the range. That's really important, I think. 
Because changing the range really impacts  your, you know, your activations quite a lot. 
So here's what that looks like. And so as you can see now, all of the backgrounds   have that nice black and it's still giving me random pixels. 
And I can check, and because I've done the  clamping, you know, and stuff, the mean and 
standard deviation aren't quite 0,  1, but they're very, very close.  So I'm going to call that good enough.  And of course the min and max haven't changed  because I clamped them to ensure they didn't 
change. So that's my random erasing.  So that randomly erases one block. And so I could create a random erase,  
which will randomly choose up to, in this case, four blocks. 
So with that function, oh, that's annoying. It happened to be zero this time. 
Okay, we'll just run it again. This time it's got three, so that's good.  So you can see it's got, oh, maybe it's  four, one, two, three, four blocks. 
Okay. So that's what this data augmentation looks like. 
So we can create a class to  do this data augmentation.  So you'll pass in what percentage to do in  each block, what the maximum number of blocks 
to have is, store that away. And then in the forward, we're just going   to call our random arrays function, passing in the input and passing in the parameters. 
Great. So now we can use   random crop, random flip and random RandErase. 
Make sure it looks okay. And so now we're going   to go all the way up to 50 epochs. And so if I run this for 50 epochs,  
I get 94.6 Isn't that crazy? 
So we're really right up there  now, up, we're even above this one. 
So we're somewhere up here. And this is like stuff people   write papers about from 2019, 2020. Oh look, here's the random erasing paper. 
That's cool. So they were way ahead of their time in 2017,  
but yeah, that would have changed for a lot longer. 
Random Copying
Now I was having a think and I realized something,  
which is like, why, like, how do I, how do we actually get the correct distribution? 
Right? Like in some ways it shouldn't matter,   but I was kind of like bothered by this thing of like, well, we don't actually end up with 0,  
1 and this kind of like clamping. It all feels a bit weird.  Like how do we actually replace these pixels  with something that is guaranteed to be the 
correct distribution? And I realized there's actually a very simple   answer to this, which is we could copy another part of the picture over to here. 
If we copy part of the picture, we're guaranteed  to have the correct distribution of pixels. 
And so it wouldn't exactly  be random erasing anymore.  That would be random copying. Now I'm sure somebody else has invented this. 
I mean, you know I'm not saying this,  nobody's ever thought of this before.  So if anybody knows a paper that's  done this, please tell me about it. 
But I, you know, I think it's a very sensible  approach and it's very, very easy to implement. 
So again, we're going to  implement it all manually, right?  So let's get our x mini batch and  let's get our, again, our size. 
And again, let's get the x, y that we're going  to be erasing, but this time we're not erasing, 
we're copying, so we'll then randomly  get a different x, y to copy from.  And so now it's just, instead of init random  noise, we just say, replace this slice of 
the batch with this slice of the batch.  And we end up with, you know, you can see  here, it's kind of copied little bits across. 
Some of them you can't really see at all. And some of you can, because I think some   of them are black and it's replaced black, but I guess it's knocked off the end of this shoe,  
added a little bit extra here, a little bit extra here.  So we can now, again, we'll  turn it into a function. 
Once I've tested it in the REPL,  make sure the function works.  And obviously this, in this case, it's copying  it largely from something that's largely black 
for a lot of them. And then again, we can do the   thing where we do it multiple times. And here we go. 
It's got a couple of random copies. And so again, turn that into a class,  
create our transforms. And again, we, okay.  So again, we can have a look at a  batch to make sure it looks sensible. 
And do it for, just did it for  25 epochs here and gets to 94% 
Now why did I do it for 25 epochs? Because I was trying to think about how   do I beat my 50 epoch record, which was 94.6 And I thought, well, what I could do is I  
Ensembling
could train for 25 epochs and then I'll train a whole new model for a different 25 epochs. 
And I'm going to put it in a  different learner, learn two, right?  This one is 94.1 So one of the models was 94.1 
One of them was 94 Maybe you can guess what we're going to do next.  It's a bit like test time augmentation, but  rather than that, we're going to grab the 
predictions of our first learner and grab  the predictions of our second learner and 
stack them up and take the mean. And this is called ensembling. 
And not surprisingly, the ensemble is better  than either of the two individual models at 
Although, unfortunately, I'm afraid  to say we didn't beat our best. 
But it's a useful trick and  particularly useful trick. 
In this case, I was kind of like trying something  a bit interesting to see if using the exact  same number of epochs, can I get a better  result by using ensembling instead of training 
for longer? And the answer was I couldn't.  Maybe it's because the random copy is not as  good, or maybe I'm using too much augmentation. 
Who knows? But it's something that you could experiment with. 
So Sharwon mentions in the chat that cutmix  is similar to this, which is actually, that's 
a good point. I'd forgotten cutmix, but cutmix, yes,   copies it from different images rather than from the same image. 
But yeah, it's pretty much  the same thing, I guess-ish.  Well, similar. Yeah, very similar. 
Wrap-up and homework
All right. So that brings us to the end of the lesson.  And I am so pumped and excited to share this  with you because I don't know that this has 
ever been done before, to be able to go from,  I mean, even in our previous courses, we've 
never done this before, go from scratch, step  by step to an absolute state of the art model 
where we build everything  ourselves and it runs this quickly.  And we're even using our own  custom ResNet and everything,  
just using common sense at every stage.  And so hopefully that shows  that deep learning is not  
magic, that we can actually build the pieces ourselves. 
And yeah, as you'll see, going up to larger  data sets, absolutely nothing changes. 
And so it's exactly these techniques. And this is actually, I do 99% of my research on  
very small data sets because you can iterate much more quickly,   you can understand them much better. And I don't think there's ever been a  
time where I've then gone up to a bigger data set and my findings didn't continue to hold true. 
Now, homework, what I would really like you to  do is to actually do the thing that I didn't 
do, which is to do the… create your own  schedulers that work with Python's optimizers. 
So, I mean, the tricky bit will be making  sure that you understand the PyTorch API well, 
which I've really laid out here. So study this carefully.  So create your own cosine  annealing scheduler from scratch,  
and then create your own 1-Cycle scheduler from scratch. 
And make sure that they work correctly  with this batch scheduler callback. 
This will be a very good exercise for you  in, you know, hopefully getting extremely 
frustrated as things don't work the way you  hoped they would and being mystified for a 
while and then working through it, you know,  using this very step-by-step approach, lots  of experimentation, lots of  exploration, and then figuring it out. 
That's the journey I'm hoping you have. If it's all super easy and you get it first go,  
then, you know, you have to find something else to do.  But yeah, I'm hoping you'll find it actually,  you know, surprisingly tricky to get it all 
working properly and in the process of doing so,  you're going to have to do a lot of exploration  and experimentation. But you'll realize that it requires no  
prerequisite knowledge at all. Okay, so if it doesn't work first time,  
it's not because there's something that you didn't learn in graduate school,   if only you had done a PhD, whatever. It's just that you need to dig through,  
you know, slowly and carefully to see how it all works.  And you know, then see how neat  and concise you can get it. 
And the other homework is to try and beat me. I really, really want people to beat me. 
Try to beat me on the 5 epoch or the  20 epoch or the 50 epoch Fashion-MNIST. 
Ideally, using miniai with things  that you've added yourself. 
But you know, you can try grabbing  other libraries if you like.  Well, ideally, if you do grab another library  and you find you can beat my approach, try 
to re-implement that library. That way you are still within the spirit  
of the game. Okay, so in our next lesson,  
Johno and Tanishq and I are going  to be putting this all together 
to create a diffusion model from scratch. And we're actually going to be taking  
a couple of lessons for this. Not just a diffusion model, but   a variety of interesting generative approaches. So we're kind of starting to come full circle. 
So thank you so much for joining  me on this very extensive journey. 
And I look forward to hearing  what you come up with.  Please do come and join us on  forums.fast.ai and share your progress. 
Bye!

---

# 19

JEREMY: Okay. Hi everybody.  And this is Lesson 19 with extremely  special guests, Tanishq and Johno. 
Hi guys. How are you?  TANISHQ: Hello. JOHNO: Hey Jeremy.  Good to be here. JEREMY: And it's New Year's Eve, 2022, finishing  
off 2022 with a bang, or at least a really cool lesson. 
And most of this lesson is  going to be Tanishq and Johno,  
but I'm going to start with a quick update from the last lesson. 
What I wanted to show you is that Christopher  Thomas on the forum, what I want to show you 
is that Christopher Thomas on the forum came up  with a better winning result for our challenge, 
the Fashion-MNIST challenge,  which we are tracking here. 
And be sure to check out this forum  thread for the latest results. 
And he found that he was able to  get better results with Dropout. 
Then Piotr on the forum  noticed I had a bug in my code. 
And the bug in my code for ResNets, actually  I won't show you, I'll just tell you, is that 
in the ResBlock, I was not passing  along the BatchNorm parameter.  And as a result, all the results  I had were without BatchNorm. 
So then when I fixed BatchNorm and added  Dropout at Christopher's suggestion, I got 
better results still. And then Christopher came up with a better   Dropout and got better results still for 50 epochs. 
So let me show you the 93.2  for 5 epochs improvement. 
I won't show the change to BatchNorm because  that's actually, that'll just be in the repo  now. So the BatchNorm is already fixed. 
Dropout
So I'm going to tell you about what  Dropout is and then show that to you.  So Dropout is a simple but powerful idea where  what we do with some particular probability, 
so here that's a probability of 0.1,  we randomly delete some activations. 
And when I say delete, what I actually  mean is we change them to zero. 
So one easy way to do this is to  create a binomial distribution object  
where the probabilities are 1-p and then sample from that. 
And that will give you a 0.1 probability. So in this case, oh, this is perfect. 
I have exactly one 0. Of course, randomly,   that's not always going to be the case. But since I asked for 10 samples and 0.1 of  
the time it should be zero, I so happened to get, yeah, exactly one of them.  And so if we took a tensor like this  and multiplied it by our activations,  
that will set about a 10th of them to zero because   multiplying by zero gives you zero. So here's a Dropout class. 
So you pass it and you say what probability  of Dropout there is, store it away.  Now we're only going to do  this during training time. 
So at evaluation time, we're not  going to randomly delete activations. 
But during training time, we will  create our binomial distribution object. 
We will pass in the 1-p probability.  And then you say, how many  binomial trials do you want to run? 
So how many coin tosses or dice  rolls or whatever each time?  And so it's just one. And this is a cool little trick. 
If you put that one onto your accelerator, you  know, GPU or MPS or whatever, it's actually 
going to create a binomial  distribution that runs on the GPU.  That's a really cool trick that  not many people know about. 
And so then if I sample and I make a sample  exactly the same size as my input, then that's 
going to give me a bunch of ones and zeros  and a tensor, the same size as my activations. 
And then another cool trick is this is going  to result in activations that are on average 
about one tenth smaller. So if I multiply by 1/(1-0.9),  
so multiply this case by that, then that's going to scale up my to undo that difference. 
JOHNO: Jeremy. JEREMY: Yeah.  JOHNO: In the line above where you have  probs equals 1-p, should that be 1-self.p 
JEREMY: Oh, it absolutely should. Thank you very much, Johno. 
Not that it matters too much because,   yeah, you can always just use nn.Dropout at this point and I only have to use 0.1,  
which is why I didn't even see that. So as you can see, I'm not even bothering   to export this because I'm just showing how to repeat what's already available in PyTorch. 
So yeah, thanks, Johno. That's a good fix.  Yeah, so if we're in evaluation mode,  it's just going to return the original. 
If p=0, then these are all  going to be just ones anyway. 
So we'll be multiplying by 1 divided  by 1, so there's nothing to change.  So with p of 0, it does nothing in effect. Yeah, and otherwise it's going to kind  
of zero out some of our activations. So we can, a pretty common place to add  
dropout is before your last linear layer. So that's what I've done here. 
So yeah, if I run the exact same epochs, I  get 93.2, which is a very slight improvement. 
And so the reason for that is that it's not  going to be able to kind of memorize the data 
or the activations, you know, because  there's a little bit of randomness.  So it's going to force it to try to identify  just the actual underlying differences. 
There's a lot of different  ways of thinking about this.  You can almost think of it as a bagging  thing, a bit like a random forest. 
You know, it's each time it's giving a  slightly different kind of random subset. 
Yeah, but that's what it does. I also added a Dropout2d layer right at the start,  
which is not particularly common. I was just kind of like showing it.  This is also how Christopher Thomas's idea tried  it as well, although he didn't use Dropout2d. 
What's the difference between  Dropout2d and Dropout?  So this is actually something I'd like you  to do to implement yourself as an exercise, 
is to implement Dropout2d. The difference is that with Dropout2d,  
rather than using x.size()  as our tensor of ones and 
zeros, so in other words, potentially dropping  out every single batch, every single channel,  every single x, y independently. Instead, we want to drop out an entire  
kind of grid area, all of the channels together. So if any of them are zero, then they're all zero. 
So you can look up the docs for Dropout2d for  more details about exactly what that looks 
like. But yeah, so the exercise is to try and implement   that from scratch and come up with a way to test it. 
So like actually check that  it's working correctly,   because it's a very easy thing to think that it's working and then realize it's not. 
So then, yeah, Christopher Thomas actually  found that if you remove this entirely and 
only keep this, then you end up  with a better results for 50 epochs.  And so he's the first to break 95% So I feel like we should insert some kind of  
animation or trumpet sounds or something at this point.  I'm not sure if I'm clever enough to do that  in the video editor, but I'll see how I go. 
Hooray! Okay.  So that's about it for me. Did you guys have any other things to add  
about Dropout, how to understand it or what it does or interesting things?  Oh, I did have one more thing before. But you go ahead if you've  
got anything to mention. JOHNO: So I was going to ask just because   I think the standard is to set it, like remove the dropout before you do inference. 
But I was wondering if there's anyone you  know of, or if it works to use it for some  sort of test time augmentation. JEREMY: Oh, dude! 
Thank you. Because I wrote a callback for that.  Did you see this or are you just like (JOHNO:  no), okay, just a test time dropout callback. 
Nice. So yeah,   before_epoch, if you're a member in  Learner, we put it into training mode. 
Which actually what it does is it puts  every individual layer into training mode.  So that's why for the module itself, we can  check whether that module's in training mode. 
So what we can actually do is after that's  happened, we can then go back in this callback 
and apply a lambda that says  if this is a Dropout, then…  
wait, this is, yeah, then put it in training mode all the time,  
including at evaluation. And so then you can run it multiple times,  
just like we did for TTA, but with this callback. Now that's very unlikely to give you a better  
result because it's not kind of showing it different versions or anything like that,  
like TTA does that are kind of meant to be the same.  But what it does do is it gives  you a sense of how confident it is. 
If it kind of has no idea, then that little  bit of dropout's quite often going to lead 
to different predictions. So this is a way of kind of doing   some kind of confidence measure. You'd have to calibrate it  
by, kind of, looking at things  that it should be confident about  and not confident about and seeing how  that dropout, test time dropout changes. 
But the basic idea, it's been  used in medical models before. 
I wouldn't say it's totally popular, which  is why I didn't even bother to show it being 
used, but I just want to add it here because  I think it's an interesting idea and maybe 
could be more used than it is, or at  least more studied than it has been. 
A lot of stuff that gets used in the medical  world is less well known out in the rest of 
the world. So maybe that's part of the problem. 
Cool. All right.  So I will stop my sharing and we're going to  switch to Tanishq, who's going to do something 
DDPM from scratch - Paper and math
much more exciting, which is to show that  we are now at a point where we can do DDPM 
from scratch or at least  everything except the model.  And so to remind you, DDPM doesn't have the  latent VAE thing and we're not going to do 
conditional.  So it's not going to be like, we're not  going to get to tell it what to draw. 
And the U-Net model itself is the  one bit we're not going to do today. 
We're going to do that next lesson. But, but other than the U-Net, it's going  
to be unconditional DDPM from scratch. So Tanishq, take it away. 
Okay. Hi, welcome back.  Sorry for the slight continuity problem. You may notice people look a little bit different. 
That's because we had some Zoom issues. So we have a couple of days  
have passed and we're back again. And then Johno over recorded his bit before   we do Tanishq's bit, and then we're going to post them in backwards. 
So hopefully there's not too many confusing  continuity problems as a result and it all  goes smoothly, but it's time to turn  it over to Tanishq to talk about DDPM. 
TANISHQ: So we've reached the point where we have  this miniai framework and I guess it's time to 
now start using it to build more,  I guess, sophisticated models. 
And as we'll see here, we can start putting  together a diffusion model from scratch using 
the miniai library, and we'll see  how it makes our life a lot easier.  And also it'd be very nice to see how the  equations in the papers correspond to the 
code. I have here,   of course, the notebook  that we'll be watching from. 
The paper, which we have the diffusion model  paper, “Denoising Diffusion Probabilistic 
Models”, which is the paper  that was published in 2020.  It was one of the original diffusion model  papers that set off the entire trend of diffusion 
models and is a good starting point  as we delve into this topic further. 
And also I have some diagrams and  drawings that I will also show later on. 
But yeah, basically let's just get started  with the code here and of course the paper. 
So just to provide some context with this  paper, this paper was published from this 
group in UC Berkeley, I think a few of  them have gone on now to work at Google. 
And this is Pieter Abbeel, he  has a big lab at UC Berkeley. 
And so diffusion models were actually originally  introduced in 2015, but this paper in 2020 
greatly simplified the diffusion models and  made it a lot easier to work with and got  these amazing results as you can see here  when they trained on faces and in this case 
CIFAR-10 and this really was very, kind  of a big leap in terms of the progress of  
diffusion models.  And so just to kind of briefly  provide, I guess, kind of an overview. 
JEREMY: If I could just quickly step  just mention something, which is,  
when we started this course, we talked a bit about how perhaps the diffusion  
part of diffusion models is not actually all that.  Everybody's been talking about  diffusion models because that's,  
particularly because that's the open source thing   we have that works really well. But this week, actually a model that appears  
to be quite a lot better than Stable Diffusion was released that doesn't use diffusion at all. 
Having said that, the basic ideas, like most  of the stuff that Tanishq talks about today, 
will still appear in some kind of form,  but a lot of the details will be different. 
But strictly speaking, actually, I don't even  know if we've got a word anymore for the kind 
of like modern generative  model things we're doing.  So in some ways, when we're talking about  diffusion models, you should maybe replace 
it in your head with some other word, which  is more general and includes this paper that  Tanishq is looking at here. JOHNO: Iterative Refinement, perhaps? 
That's what I'd like. JEREMY: Yeah, that's   not bad, iterative refinement. I'm sure by the time people watch this video,  
probably, you know, somebody will have decided on something.  We will keep our course website up to date. TANISHQ: Yeah. 
Yeah. This is the paper that Jeremy was talking   about and yeah, every week there seems to be another state of the art model. 
But yeah, like Jeremy said, a lot of the  principles are the same, but the details   can be different for each paper. 
And I just want to again, also, like Jeremy  was saying, zoom back a little bit and talk 
a little bit about what, just to provide  a review of what we're trying to do here. 
So let me just right next to him here. Yeah.  So with this task, we were trying to, in this  case, we're trying to do image generation. 
Of course, it could be other forms of  generation, like text generation or whatever.  And the general idea is that of  course we have some data points. 
In this case, we have some images of dogs  and we want to produce more like the data  points that we're given. So in this case, maybe the  
dog image generation or something like this. And so the overall idea that a lot of these  
approaches take for some sort of generative modeling task is they try to... 
Not over there, I’m going to mark here. They try to... 
Oops, what happened here? Maybe it might...  Yeah. So let me use it in a bit. 
p(x), which is basically the likelihood... What's going to happen here? 
Likelihood of data point x. So let's say x is some image. 
Then p(x) tells us what is the probability  that you would see that image in real life. 
And we can take a simpler example, which may  be easier to think about, of a one-dimensional 
data point like height, for example.  And if we were to look at height, of course  we know we have a data distribution that's 
kind of a bell curve. And you have maybe some mean height,  
which is something like 5'9", 5'10". I guess 5'10", or something like that,  
or 5'9", whatever. And then of course we have some more   unlikely points, but that is still possible. Like for example, we have 7'8", or we have  
something that's maybe not as likely, which is like 3', or something like this.  JEREMY: So here's the X axis is height, and the  Y axis is the probability of some random person 
you meet being that tall. TANISHQ: Exactly.  So this is basically the probability. And so of course you have this sort of peak,  
which is where you have higher probability. And so those are the sorts of values   that you would see more often. So this is what we would call our p(x). 
And the important part about p(x) is that  you can use this now to sample new values 
if you know what p(x) is, or if you have  some sort of information about p(x).  So for example, here you can think of, if  you were to say, maybe have some, let's say 
you have some game and you have some human  characters in the game, and you just want  to randomly generate a height for this human  character, you wouldn't want to of course 
select a random height between 3 and 7,  that's kind of uniformly distributed.  You would instead maybe want to have the height  dependent on this sort of function, where 
you would more likely sample values in the  middle and less likely sample these sorts 
of extreme points. So it's dependent on this function p(x).  So having some information about p(x)  will allow you to sample more data points. 
And so that's kind of the overall goal of  generative modeling is to get some information  about p(x) that then allows us to sample  new points and create new generations. 
So that's kind of a high level kind of description  of what we're trying to do when we're doing 
generative modeling. And of course there are many different approaches.  We have our famous GANs, which used to be the  common method back in the day before diffusion 
models. We have VAEs,   which I think we'll probably talk a  little bit more about that later as 
well. JEREMY: We'll be talking about   both of those techniques later. Yeah.  TANISHQ: Yeah. So there are many different other techniques. 
There are also some niche techniques  that are out there as well.  But of course now the popular one is are these  diffusion models or as we talked about, maybe 
a better term might be, iterative  refinement or whatever the term ends to be. 
But yeah, so there are many different techniques. And yeah, so this is kind of the general  
diagram that shows what diffusion models are. And if we can look at the paper here,  
which let's pull up the paper. Yeah, you see here, this is the sort of,  
they call it directed graphical model. It's a very complicated term.  It's just kind of showing  what's going on in this process. 
There's a lot of complicated math here, but  we'll highlight some of the key variables 
and equations here. So basically the   idea is that, okay, so let's see here. This is an image that we want to generate, right? 
And so x0 is basically, these are  actually the samples that we want. 
So we want to, x0 is what we want to generate. And these would be, yeah, these are images. 
And we start out with pure noise. So that's what xt, pure noise. 
And the whole idea is that we have two processes. We have this process where we're going from pure  
noise to our image. And we have this   process from our image to pure noise. So the process where we're going from our image   to pure noise, this is called the forward process. 
Forward, sorry, my typing is still,  my handwriting is not so good in it.  So hopefully it's clear enough. Let me know if it's not. 
So we have the forward process, which  is mostly just used for training.  Then we also have our reverse process. This is the reverse process, which  
I will write up here. Reverse process.  JEREMY: So this is a bit of a summary, I guess,  of what you and Wasim talked about in Lesson 
TANISHQ: And just, it's just mostly to highlight  now what are the different variables as we look 
at the code and see the  different variables in the code.  JEREMY: Okay, so we'll be focusing today on the  code, but the code will be referring to things by 
name and those names won't make sense very  much unless we see what they're used for in 
the math. Okay.  TANISHQ: Yeah. And I won't dive too much into the math.  I just want to focus on these sorts of  variables and equations that we see in the code. 
So basically the general idea is that  we do these in multiple different steps. 
We have here from time step 0 all the way to  time step uppercase T. And so there's some 
fixed number of steps, but then we have this  intermediate process where we're going from 
some particular time step. We have this time step  
lowercase t, which is some noisy image. And yes, we're transitioning between  
these two different noisy images. So we have this, what is sometimes   called the transition. We have this one here. 
This is like something that's  called the transition kernel or   yeah, whatever it is, it basically is just telling us, you know, how do we go  
from, you know, one, in this case, we're going from a less noisy image to a more noisy image  
and then going backwards, it's going from a more noisy image to a less noisy image.  So let's look at the equations. JEREMY: So the forward direction is driven  
really easily to make it something more noisy. You just add a bit more noise to it.  And the reverse direction is incredibly difficult,  which is to particularly to go from the far 
left to the far right is strictly  speaking impossible because none of that  
person's face exists anymore.  That somewhere in between you could certainly  go from something that's partially noisy to 
less noisy by a learned model. TANISHQ: Exactly. 
And that's like one of the little things I've  done right now in terms of, you know, in terms  of I guess the symbols and the math.  So yeah, basically I'm just trying to pull  out the, just to write down the equations 
here. So we have, let me zoom in a bit. 
So we have our two, let's see here. Q of xt, x (t minus 1) 
Or actually, you know what, maybe it's  just better if I just snipped it from here. 
So the one that is going from our  forward process is this equation here. 
So I'll just make that a  little smaller for you guys.  So right there. So that is going, and basically to explain,  
we have this sort of script, a little bit of a, maybe a little bit confusing notation  
here, but basically this is referring to a normal distribution or a Gaussian distribution. 
And this is just saying, okay, this is a Gaussian  distribution that's describing this particular  variable. So it's just saying,  
okay, N is our normal or Gaussian  distribution, and it's representing 
this variable x of t, or x, sorry,  xt. And then we have here is the mean,  
and this is the variance. 
So just to again, clarify, I think we've talked  about this before as well, but this is a, 
this is of course a bad drawing of a Gaussian,  but our mean is just, our mean is just this, 
the middle point here is the mean, and the  variance just kind of describes the sort of  spread of the Gaussian distribution. So if you think about it a little further,  
you have this beta, which is one of the important  variables that kind of describes  the diffusion process, beta-t 
So you'll see the beta-t in the code. And basically beta-t increases as t increases. 
So basically your beta-t will be  greater than your beta-(t minus 1). 
So if you think about that a little bit more  carefully, you can see that, okay, so at t-1, 
at this time point here, and then you're  going to the next time point, you're  going to increase your beta-t. So you're increasing the variance,  
but then you have this 1 - beta-t and take the square root of that and multiply  
it by x(t minus 1) So as your t is increasing,   this term actually decreases. So your mean is actually decreasing  
and you're getting less of the original image Because the original image is going to be part  
of x(t minus 1) JEREMY: And just to let you  
know, Tanishq, we can't see your pointer. So if you want to point at things,  
you would need to highlight them or something. TANISHQ: So yeah, I'll just, let's see. 
Yeah. Basically, I haven't pointed anything in specific. 
I was just saying that, yeah, basically  if we have our xt here, as the time step 
increases, you're getting less  contribution from your x(t minus 1) 
And so that means your mean is going towards zero. And so you've got to have a mean of 0 and  
the variance keeps increasing and basically you just have a Gaussian distribution and you   lose any contribution from the original image as your time step increases. 
So that's why when we start out from  x0 and go all the way to our xt here, 
this becomes pure noise. It's because we're doing   this iterative process where we keep adding noise. We lose that contribution from the original image  
and that leads to the image having pure noise at the end of the process. 
JEREMY: Something I find useful here is to  consider one extreme, which is to consider x1. 
So at x1, the mean is going to  be root 1 - beta-t times x0. 
The reason that's interesting  is x0 is the original image.  So we're taking the original  image and at this point  
1 - beta-t will be pretty close to 1.  So at x1, we're going to have something  that's the mean is very close to the image 
and the variance will be very small. And so that's why we will have a  
image that just has a tiny bit of noise. TANISHQ: Right, right. 
And then another thing that sometimes it's  easier to write out is sometimes you can write  out, in this case, you can write out q(xt)  directly because these are all independent 
in terms of q(xt) is only  dependent on x(t minus 1)  And then x(t minus 1) is only  dependent on x(t minus 2) 
And each of these steps are independent. So based on the different laws of probability,  
you can get your q(xt) in close form. So, yeah, that's what's shown here. 
q xt given the original image. So this is also another way of kind of seeing   this more clearly where you can see that. Anyways, so I'm going back here. 
So this is another way to see here more directly. So this is, of course, our clean image. 
And this is our clear, our noisy image. And so you can also see again,  
now alpha bar t is dependent on beta t. Basically it's like one minus the cumulative. 
JEREMY: I mean, we'll see  the code for it, I guess.  So maybe. TANISHQ: Yes, yes. 
So it might be clear to see that this  is alpha bar t or something like this.  But basically, basically the idea is that  alpha bar t is going to be, again, less. 
This is what is going to be  less than alpha bar t minus one.  So basically alpha, this keeps decreasing, right? This decreases as time step increases. 
And on the other hand, this is going to  be increasing as time step increases.  But again, you can see the contribution from  the original image decreases as time step 
increases while the noise, as shown by the  variance, is increasing while the time step 
is increasing. Anyway, so that   hopefully clarifies the forward process. And then the reverse process is basically  
a neural network, as Jeremy had mentioned. And yeah,  
screenshot this. That's our reverse process. 
And basically the idea is, well, this is a  neural network and this is also a neural network. 
Neural network. And we learn it during the training of the model. 
But the nice thing about this particular diffusion  model paper that made it so simple was actually, 
we completely ignored this and actually set  it to constants just based on, you know, big 
numbers. JEREMY: We can't see what you're pointing at.  So I think it's important to  mention what this is here.  TANISHQ: This term here. So this one, we just kind of ignore  
and it's just a constant dependent on beta-t. So you only have one neural network that you need  
to train, which is basically referring to this mean.  And when the nice thing about this diffusion  model process is that it also re-paraphrases 
the mean into this easier form where  you do a lot of complicated math,  
which we'll not get into here.  But basically you get this kind of simplified  training objective where, let's see here. 
Yeah, you see the simplified training objective. You instead have this epsilon-theta function. 
And let me just screenshot that again. 
This is our loss function that we train  and we have this epsilon-theta function. 
You can see it's a very  simple loss function, right?  This is just a, let me just write this down. This is just an MSE loss. 
And we have this epsilon-theta function here. That is our-  JEREMY: …maybe here we're less mathy, it  might not be obvious that it's a simple 
thing, because it looks quite complicated  to me, but once we see it in code, it'll be 
simple. TANISHQ: Yes, yes.  Basically you're just doing like, and  you'll see it in code how simple it is. 
But this is like just an MSE loss. So we've seen MSE loss before,   but you'll see how, yeah, this is basically MSE. 
So the nice, so just to kind of take a step  back again, what is this epsilon-theta? 
Because this is like a new thing that  like seems a little bit confusing.  Basically epsilon, you can see here, basically,  yeah, so this here is saying, this is actually 
equivalent to this equation here. These two are equivalent. 
This is just another way of saying that,  because basically it's saying, that's xt 
So this is giving xt just in a different way. But epsilon is actually  
this normal distribution with a  mean of 0 and a variance of 1. 
And then you have all these scaling terms  that changes the mean to be the same as this 
equation that we have over here. So this is our xt. And so what epsilon is,  
it's actually the noise that we're adding to our image to make it into a noisy image.  And what this neural network is doing  is trying to predict that noise. 
So what this is actually doing is this is  actually a noise predictor, and it is predicting 
the noise in the image. And why is that important? 
Basically the general idea is,  if we were to think about our  
distribution of data, let's just think about it in a 2D space. 
Just here, each data point here represents an  image, and they're in this blob area, which 
represents a distribution. So this is in-distribution,  
and this is out of the distribution. Out-of-distribution. 
And basically the idea is that, okay, if we  take an image and we want to generate some 
random image, if we were to take a random  data point, it would most likely be noisy 
images, right? So if we take some   random data point, it would be more,  the way to generate random data point, 
it's going to be just noise. But we want to keep adjusting this data point to  
make it look more like an image from your distribution.  That's kind of the whole idea of this iterative  process that we're doing in our diffusion 
model. So the way to get that information is actually   to take images from your dataset and actually add noise to it. 
So that's what we try to do in this process. So we have an image here and we add noise to it. 
And then what we do is we try to plan  a neural network to predict the noise.  And by predicting the noise and subtracting  it out, we are going back to the distribution. 
So adding the noise takes you away from the  distribution and then predicting the noise  brings you back to the distribution. So then if we know at any given point in this  
space how much noise to remove, that tells you how to keep going towards the data  
distribution and get a point that  lies within the distribution. 
So that's why we have noise prediction. And that's the importance of doing   this noise prediction is to be  able to then do this iterative 
processor, we can start out at a random point,  which would be, for example, pure noise and  keep predicting and removing that noise  and walking towards the data distribution. 
DDPM - The code
Okay. Okay.  So yeah, let's get started with the code. And so here we of course have our imports  
and we're going to load our dataset. We're going to work with our Fashion-MNIST  
dataset, which is what we've been working with for a while already.  And yeah, this is just basically the same  code that we've seen from before in terms 
of loading the dataset.  And then we have our model. So I've removed the noise from an image. 
So our model is going to take in, is it's  going to take in the previous image, the noisy 
image and predict the noise. So the shapes of the   input and the output are the same. They're going to be in the shape of an image. 
U-Net Neural Network
So what we use is we use a U-Net neural  network, which takes in kind of an input image. 
JEREMY: And we do see your  pointer now, by the way.  So feel free to point at things. TANISHQ: Yeah. 
So yeah, it takes in an input image. And in this case, a U-Net  
is the purpose, but they can also  be used for any sort of image  to image task, where we're going from an input  image and then outputting some other image 
of some sort. And we'll talk about...  JEREMY: So this is a new architecture, which we  haven't learned about yet, and we will be learning 
about in the next lesson. But broadly speaking, those gray arrows  
going from left to right are a lot like ResNet, very much like ResNet skip connections. 
But they're being used in a different way. Everything else is stuff that we've seen before. 
So it's basically, we can pretend  those don't exist for now.  It's a neural network that the output is the  same size or a similar size to the input. 
And therefore you can use it to learn how  to go from one image to a different image. 
TANISHQ: Yeah. So that's where the U-Net is.  And yeah, like Jeremy said,  we'll talk about it more. 
The sort of U-Net that are used for diffusion  models also tend to have some additional tricks, 
which again, we'll talk  about them later on as well.  But yeah, for the time being, we will just  import a U-Net from the Diffusers library, 
which is the Hugging Face  library for diffusion models.  So they have a U-Net implementation  and we'll just be using that for now. 
And so, yeah, of course,  JEREMY: strictly speaking,   we're cheating at this point because we're using something we haven't written from scratch,  
but we're only cheating temporarily because we will be writing it from scratch.  TANISHQ: Yeah. And yeah, so, and then of course we're  
working with one channel images, our Fashion-MNIST images are one channel images.  So we just have to specify that. And then of course the channels of the different  
blocks within the U-Net are also specified. And then let's go into the training process. 
Training process
So basically the general idea of course  is we want to train with this MSE loss. 
What we do is we select a random timestep  and then we add noise to our image based on 
that timestep. So of course, if we have a very high timestep,   we're adding a lot of noise. If we have a lower timestep,  
they were adding very little noise. So we're going to randomly choose a timestep. 
And then, yeah, we add the  noise accordingly to the image. 
And then we pass the noisy image  to a model as well as the timestep.  And we are trying to predict the amount of  noise that was in the image and we predict it 
with the MSE loss. So we can see all the...  JEREMY: I have some pictures of some of these  variables I could share if that would be useful. 
So I have a version. So I think Tanishq is sharing notebook number 15.  Is that right? And I've got here notebook number 17. 
And so I took Tanishq's notebook and just,  as I was starting to understand it, I  like to draw pictures for myself  to understand what's going on. 
So I took the things which are in Tanishq's  class and just put them into a cell. 
So I just copied and pasted them, although  I replaced the Greek letters with English 
written out versions. And then I just plotted   them to see what they look like. So in Tanishq's class, he has this thing  
called beta, which is just linspace. So that's just literally a line. 
So beta, there's going to be a thousand of  them and they're just going to be equally  spaced from 0.0001 to 0.02. And then there's something called sigma,  
which is the square root of that. So that's what sigma is going to look like. 
And then he's also got alphabar, which  is a cumulative product of 1 minus this. 
And there's what alphabar looks like. So you can see here, as Tanishq was describing  
earlier, that when t is higher, this is t on the x-axis, beta is higher,  
and when t is higher, alphabar is lower. So yeah, so if you want to remind yourself,  
so each of these things, beta, sigma, alphabar, they're each,  
they've each got a thousand things in them. And this is the shape of those thousand things. 
So this is the amount of variance,  I guess, added at each step. 
This is the square root of that. So it's the standard deviation added at each step. 
And then if we do 1 minus that,  it's just the exact opposite. 
And then this is what happens if you  multiply them all together up to that point.  And the reason you do that is because if you add  noise to something, you add noise to something 
that you add noise to something that you add  noise to something, then you have to multiply  together all that amount of noise  to say how much noise you would get. 
So yeah, those are my pictures, if that's helpful. TANISHQ: Yep. 
Yep. Good to see the diagram or see how the   actual values and how it changes over time. So yeah. 
Let's see here. Sorry.  Yeah. So like Jeremy was showing,  
we have our linspace for our beta. In this case, we're using kind   of more of the Greek letters. So you can see the Greek letters  
that we see in the paper as well as... Now we have it here in the code as well. 
And we have our linspace from our  minimum value to our maximum value.  And we have some number of steps. So this is the number of timesteps. 
So here we use a thousand timesteps, but  that can depend on the type of model that 
you're training. And that's one of the parameters   of your model or hyperparameters of your model. JEREMY: And this is the callback you've got here. 
So this callback is going to be used to set  up the data, I guess, so that you're going 
to be using this to add the noise so that the  model's then got the data that we're trying 
to get it to learn to then denoise. TANISHQ: Yeah.  So the callback of course makes life a lot  easier in terms of, yeah, setting up everything 
and still being able to use, I guess, the  miniai learner with maybe some of these more 
complicated and maybe a little  bit more unique training loops.  So yeah, in this case, we're just able to  use the callback in order to set up the batch 
that we are passing into our learner. JEREMY: I just want to mention, when you first did  
this, you wrote out the Greek letters in English, alpha and beta and so forth. 
And at least for my brain, I was finding it  difficult to read because they were literally 
going off the edge of the page  and I couldn't see it all at once.  And so we did a search and replace to  replace it with the actual Greek letters. 
I still don't know how I feel about it. I'm finding it easier to read because I  
can see it all at once. I don't know if it's a   scroll and I don't get overwhelmed. But when I need to edit the code,  
I kind of just tend to copy  and paste the Greek letters,  which is why we use the actual word beta in  the init parameter list so that somebody using 
this never has to type a Greek letter. But I don't know, Johno or Tanishq,   if you had any thoughts over the last week or two, since we made that change about whether  
you guys like having the Greek letters in there or not.  JOHNO: I like it for this demo in particular. I don't know that I do this in my code,  
but because we're looking back and forth between the paper and the implementation here, I think it  
works in this case just fine. TANISHQ: Yeah, I agree.  I think it's good for when you're trying to  study something or try to implement something, 
having the Greek letters is very useful to be  able to, I guess, match the math more closely 
and it's just easy just to pick the equation  and put it into code or white-source style 
looking at the code and try  to match it to the equation.  So I think for educational purpose, I  tend to like, I guess, the Greek letters. 
So yeah.  Yeah, so we have our initialization where  we're just defining all these variables. 
We'll get to the predict in just a moment, but  first I just want to go over the before_batch  where we're setting up our  batch to pass into the model. 
So remember that the model is taking  in our noisy image and the timestep. 
And of course the target is the actual amount  of noise that we are adding to the image. 
So basically we generate that noise. So that's what…  JEREMY: So epsilon is that target. So epsilon is the amount of noise,  
not the amount of, is the actual noise. TANISHQ: Yes.  Epsilon is the actual noise that we're adding. And that's the target as well because  
our model is a noise predicting model. It's predicting the noise in the image.  And so our target should be the noise  [unintelligible] that we're adding to  
the image during training.  So we have our epsilon and we're just  generating it with this random function. 
The random normal distribution  with a mean of 0, variance of 1.  So that's what that's doing and adding  the appropriate shape and device. 
Then the batch that we get originally  will contain the clean images. 
These are the original images from our dataset. So that's x0. 
And then what we want to  do is we want to add noise.  So we have our alphabar and we have  a random timestep that we select. 
And then we just simply follow that equation,  which I can, I'll just show in just a moment.  JEREMY: That equation, you can  make a tiny bit easier to read. 
I think if you were to double click on that  first alphabar underscore t, cut it and then 
paste it, sorry, in the xt equals torch dot  square root, take the thing inside the square 
root, double click it and paste  it over the top of the word torch. 
That would be a little bit easier to read. 
And then you'll do the same for the next one. TANISHQ:  
There we go. Put those parentheses.  Yep. TANISHQ:  
Yeah, so basically, yeah, so yeah, I  guess I'll just pull up the equation. 
So let's see, there's then, so there's a section  in the paper that has the nice algorithm. 
See if I can find it. No, no, here.  It's, I think earlier. Yes, training. 
Right, so this, we're just following these  same sort of training steps here, right?  Where we select a clean image  that we take from our data set. 
This fancy kind of equation here is just saying,  take an image from your data set, take a random 
timestep between this range. Then this is our epsilon that we're  
getting, just saying, get some epsilon value. And then we have our equation for Xt, right? 
This is the equation here. You can see that it is square root of alphabar   t x0 plus one square root of one minus alphabar t times epsilon. 
So that's the same equation  that we have right here, right?  And then what we need to do is we  need to pass this into our model. 
So we have xt and t. So we  set up our batch accordingly.  So this is the two things  that we pass into our model. 
And of course we also have our  target, which is our epsilon.  And so that's what this is showing here. We passed in our Xt as well as our t here, right? 
And we pass that into a model. The model is represented here as epsilon theta.  And theta is often used to represent, like, this  is a neural network with some parameters and 
the parameters are represented by theta. So epsilon theta is just representing   our noise predicting model. So this is our neural network. 
So we have passed in our Xt and our t into  a neural network, and we are comparing 
it to our target here,  which is the actual epsilon.  And so that's what we're doing here. We have our batch where we  
have our xt and t and epsilon. And then here we have our prediction function. 
And because we actually have, I guess in this  case, we have two things that are in a tuple 
that we need to pass into our model. So we just kind of get those  
elements from our tuple with this. We get the elements from the tuple,   pass it into the model, and then Hugging Face has its own API in terms of getting the output. 
So you'd need to call .sample in order  to get the predictions from your model.  So we just do that. And then we do, we have learn.preds and that's  
what's going to be used later then when we're trying to do our loss function calculation. 
Inheriting from miniai TrainCB
JEREMY: So the, just so, I mean, it's just worth  looking at that a little bit more since we haven't  quite seen something like this before. And it's something which I'm not aware of  
any other framework that would let you do this, you know,   literally replace how prediction works. And miniai is kind of really fun for this. 
So because you're inherited from TrainCB,  TrainCB has predict, a dot defined and 
you've defined a new version. So it's not going to use   the TrainCB version anymore. It's going to use your version. 
And what you're doing is instead of passing  learn.batch[0] to the model, you're, you've 
got a * in front of it. So the key thing is that * is going to, you know,   and is, we know that actually learn.batch[0] has two things in it because that learn.batch  
that you showed at the end of the before_batch method has two things in learn dot zero.  So that star will unpack them and send  each one of those as a separate argument. 
So our model needs to take two things, which  the diffusers U-Net does take two things. 
So that's the main interesting point. And then something I find a bit awkward honestly,  
about a lot of Hugging Face stuff, including diffusers is that generally their models don't  
just return the result, but they put it inside some name.  And so that's what happens here. They put it in something  
inside something called sample. So that's why Tanishq added .sample at the   end of the predict because of this somewhat awkward thing, which Hugging Face  
like to do for some reason. But yeah, now that you know, I mean,   this is something that people often get stuck on. I see on Kaggle and stuff like that. 
It's like, how on earth do I use these models? Because they take things in weird forms and they   give back things with weird forms. Well, this is hell. 
You know, if you inherit from TranCB, you  can change predict to do whatever you want, 
which I think is quite sweet. TANISHQ: Yep. 
So yeah, that's the training loop. And then of course you have your regular training  
loop that's implemented in miniai where you are going to have. 
Yeah. So you have your loss function calculation,   I mean, and the predictions, learn.preds. And of course the target is our learn.batch[1],  
which is our epsilon. So we have those and   we pass it into the loss function. It calculates the loss function and  
does the back propagation. So I'll just go over that.  We'll get back to the sampling in just a moment. But just to show the training loop. 
JEREMY: Most of this is copied from  our, I think it's 14_augment notebook,  
the way you've got the tmax and the sched.  The only thing I think you've added  here is the DDPM callback, right? 
TANISHQ: Yes. The DDPM callback.  JEREMY: And you change the loss function. TANISHQ: Yes. 
So basically we have to initialize our DDPM  callback with the appropriate arguments. 
So like the number of timesteps and  the minimum beta and maximum beta. 
And then yeah, obviously, and then of course  we're using an MSE loss as we talked about. 
It just becomes a regular training  loop and everything else is run before.  Yeah. So we have your scheduler,  
your progress bar, all of that we've seen before. JEREMY: I think that's really cool that we're   using basically the same code to train a diffusion 
model as we've used to train a  classifier just with one extra callback.  TANISHQ: Yeah. Yeah. 
Yeah. That's why I think callbacks are very   powerful for allowing us to do such things. It's like pretty,  
you can take all this code and now  we have a diffusion training loop  and we can just call learn.fit and yeah,  they can see got a nice training loop, nice 
loss curve. We can save our model   on a torch saving functionality  to be able to save our model  
and we could load it in.  But now that we have our trained model, then  the question is, what can we do to use it 
Using the trained model: denoising with “sample” method
to sample the dataset? So the basic idea of course was that  
we have, like basically we're here, right? We have, let's see here. 
Okay. So we have, the basic idea is that we start   out with a random data point and of course that's not going to be within the distribution at first,  
but now we've learned how to move from one point towards the data distribution. 
That's what our noise prediction,  predicting function does.  It basically tells you how, you know,  in what direction and how much to,  
so the basic idea is that, yeah, I guess I'll   start from maybe a new drawing here. Again, we have,  
distribution is, and we have a random point   and we use our noise predicting model that we have trained to  
tell us which direction to move. So it tells us some direction.  Or I guess what's the exact, other area. Okay.  So like here, okay. So it tells us some direction to move. 
At first that direction is not going to be  like, you cannot follow that direction all  the way to get the correct data point. Because basically what we were doing is we're  
trying to reverse the path that we were following when we were adding noise.  So like, cause we had originally data point  and we kept adding noise to the data point 
and maybe, you know, it  followed some path like this.  And we want to reverse that path to get to. So our noise predicting function will give  
us an original direction, which, you know, would be, you know, some kind of, it's going  
to be kind of tangential to the actual path at that location.  So what we would do is, you know, we would maybe  follow that data point all the way towards, 
you know, we're just going to  keep following that data point. 
You know, we're going to try to predict the  fully denoised image by following this noise 
prediction. But our fully denoised   image is also not going to be a real image. So what we, so let me, I'll show an example  
of that over here in the paper on why they show this a little bit more carefully. 
So x0, it's there. So basically   you can see the different data... You can see the different data  
points here. It's not going   to look anything like our real image. So you can see all these points, you know,  
it doesn't look anything… what we would do is  
we actually had a little bit of noise back to it. And we start, we have a new point where then we  
could maybe estimate a better, get a better estimate of which direction to move, follow  
that all the way again, we follow a new point. And then I can add back a little bit of noise.  You get a new estimate, you make a new estimate  of, you know, this noise prediction and removing 
the noise, you know, fall that  all again, completely and add a   little bit of noise again to the image and burst onto a image. 
So that's kind of what we're showing here. JEREMY: That's a lot like SGD, with SGD we   don't take the gradient and jump all the way. We use a learning rate to go some of the way  
because each of those estimates of where we want to go, you know, not that great,  
but we just do it slowly. TANISHQ: Exactly.  And at the end of the day, that's what  we're doing with this noise prediction. 
We are predicting the sort of gradient of  this p(x), but of course we need to keep 
making estimates of that  gradient as we're progressing.  So we have to keep evaluating our noise prediction  function to get updated and better estimates 
of our gradient in order to  finally converge onto our image.  So and then you can see that here, you know, we  have this, maybe this fully predicted denoised 
image which at the beginning doesn't look  anything like a real image, but then as we 
continue throughout the sampling  process, we finally converge on   something that looks like an actual image. 
Again, these are CIFAR-10 images and it's  still a little bit maybe unclear about how  realistic these images, these very small images  look, but that's kind of the general principle 
I would say. And so that's what I can show in the code. 
This idea of we're going to start out  basically with a random image, right? 
And this random image is going to be like  a pure noise image and it's not going to be  part of the data distribution. You know, it’s not anything like  
a real image, it's just a random image. And so this is going to be our x, I guess,  
x uppercase t [x_t], right? That's what we start out with.  And we want to go from x  uppercase t all the way to x0. 
So what we do is we go through each of the  timesteps and we create, we have to put it 
in this sort of batch format because  that's what our neural network expects.  So we just have to format it appropriately. And we'll get to z in just a moment. 
I'll explain that in just a moment, but of  course we just take it have similar… alphabar, 
betabar, which is getting  those variables that we need.  JEREMY: And we faked beta bar because  we couldn't figure out how to type it. 
So we used bbar instead. TANISHQ: Yeah.  So yeah, so yeah, in, yeah, we, yeah, we were  pretty able to get betabar to work, I guess. 
But anyway, at each step, what we're trying  to do is to try to predict what direction  we need to go. And that direction is given  
by our noise predicting model, right? So what we do is we pass in x_t and  
our current timestep into our model. And we get this noise prediction and   that's the direction that we need to move in. So basically we take x_t. We first attempt to  
completely remove the noise, right? That's what this is doing.  That's what x_0_hat is. That's completely removing the noise. 
And of course, as we said, that estimate  at the beginning won't be very accurate.  And so now what we do is we have some coefficients  here where we have a coefficient of how much 
that we keep about this estimate of  our denoise image and how much of the  
originally noisy image we keep.  And on top of that, we're going  to add in some additional noise. 
So that's what we do here. We have x_0_hat. 
And so, and we multiply by its coefficient  and we have x_t we multiply it by some 
coefficient and we also add some additional noise. That's what the z is.  It's just- JEREMY: That's basically a weighted average of  
the two plus the noise… TANISHQ: Exactly.  And then the whole idea is that as we get  closer and closer to a timestep equals to 0 
our estimate of x0 will be more and more accurate. So our x0_coeff will get closer  
as we're increasing our, going through the  process and then our xt_coeff  will get closer and closer to 0. 
So basically we're going to be weighting more  and more of the x_0_hat estimate and less 
and less of the x_t as we're getting  closer and closer to our final timestep.  And so at the end of the day, we will  have our estimated generated image. 
So that's kind of an overview  of the sampling process. 
So yeah. So yeah, basically the way I implemented  
it here was I had the sample function that's part of our callback and it will take in the  
model and the kind of shape that you want for your images that you're producing. 
So like if you want to specify how many images  you produce, that's going to be part of your  batch size or whatever. And you'll just see that in a moment. 
But yeah, it's just part of the callback. So then we basically have our DDPM callback  
and then we could just call the sample method of our DDPM callback and we pass in our model. 
Inference: generating some images
And then here you can see we're going to produce,  for example, 16 images and it just has to  be a 1 channel image of shape  32 by 32 and we get our samples. 
And one thing I forgot to note was that I  am collecting each of the timestep, the x_t. 
So the predictions here, you can see  that there are a thousand of them. 
We want the last one because  that is our final generation.  So we want the last one and that's what we should- JEREMY: They are not [sad] actually. 
TANISHQ: Yeah. So this is-  JEREMY: We've come a long way since DDPM. So this is, like,   slower and less great than it could be. But considering that, except for U-NET,  
we've done this from scratch, you know, actually from matrix multiplication,  
I think those are pretty decent. TANISHQ: Yeah.  And we're only trained for about five epochs. It took like, you know, maybe like four minutes  
to train this model, something like that. It's pretty quick.  And this is what we get with very little training. And it's pretty decent. 
You can see, of course, some clear shirts  and shoes and pants and whatever else. 
JEREMY: Yeah. And you can see fabric and it's   got texture and things have buckles and- TANISHQ: Yeah. 
JEREMY: You know, something to compare, like, we  did generative modeling in the first time we did 
Part 2 back in the days when Wasserstein  GAN was just new, which was actually 
created by the same guy that created  PyTorch or one of the two guys, Soumith.  And we trained for hours and hours and hours  and got things that I'm not sure were any 
better than this. So things have come a long way.  TANISHQ: Yeah. Yeah. 
And of course, then, yeah, so we can see then  like how this sampling progresses over time, 
over the multiple timesteps. So that's what I'm showing here because   I collected, during the sampling process, we are collecting at each timestep what  
that estimate looks like. And you can kind of see here.  And so this is an estimate of the  noisy image over the timesteps. 
Oops. And I guess I had to pause.  Yeah. You can kind of see.  But you'll notice that actually, so we actually,  what we did is like, okay, so we selected 
an image, which is like the ninth image. So that's, that's this image here.  So we're looking at this image  particularly here and we're going over. 
Yeah. We have a function here that's showing the i-th   timestep during the sampling process of that image. 
And we're just getting the images. And what we are doing is we're only  
showing basically from timestep 800 to a 1,000. And here we're just, we're just having it like  
where it's like, okay, we're looking at like maybe every 5 steps and   we're going from 800 to 990. And this time it would make  
it visually easier to see the transition. But what you'll notice is I didn't start  
all the way from 0. I started from 800.  And the reason we do that is because actually  between 0 and 800 there's very little change 
in terms of like, it's just mostly a noisy image. And it turns out, but yeah, I didn't see as I make  
a note of this here, it's actually a limitation of the noise schedule   that is used in the original DDPM paper. And especially when applied to some of these  
smaller images, when we're working with images of like size 32 by 32 or whatever. 
And so there are some other papers like the  improved DDPM paper that propose other sorts 
of noise schedules. And what I mean by noise schedule   is basically how beta is defined basically. So we had this definition of torch.linspace for  
our beta, but people have different ways of defining beta that lead   to different properties. So things like that, people have come up  
with different improvements and those sorts of improvements work well when we're   working with these smaller images. And basically the point is like, if  
we are working from 0 to 800 and it's just mostly just noise that entire time, we're not actually  
making full use of all this timesteps. So it would be nice if we could actually   make full use of those time steps and actually have it do something during that time period. 
So all these, there are some papers that examine  this a little bit more carefully and it would  be kind of interesting for maybe some of you  folks to also look at these papers and see 
if you can try to implement those sorts of  models with this notebook as a starting point. 
And it should be a fairly simple change in  terms of like noise schedule or something  like that. JEREMY: So I actually think this is the start of  
our next journey, which is our previous journey was going from being totally rubbish at  
Fashion-MNIST classification  to being really good at it.  I heard you say now we're like a little bit  rubbish at doing Fashion-MNIST generation. 
And yeah, I think we should all now work from  here over the next few lessons and so forth 
and people trying things at home and all of  us trying to make better and better generative 
models, initially a Fashion-MNIST and hopefully  we'll get to the point where we're so good  at that, that we're like, oh, this is too easy. And then we'll pick something harder. 
And eventually that'll take us to   Stable Diffusion and beyond. I imagine. 
That's cool. I got some stuff to show you guys. 
Notebook 17: Jeremy’s exploration of Tanishq’s notebook
If you're interested, I tried to better understand  what was going on in Tanishq's notebook and 
tried doing it in a thousand different ways  and also see if I could just start to make  it a bit faster. So that's what's in notebook 17,  
which I will share. So we've already seen the start of notebook 17. 
Well, the one thing I did just do is just  drew a picture for myself, partly just to  remind myself what they,  what the real ones look like. 
And they definitely have more detail than  the samples that Tanishq was showing. 
But they're not, you know, they're just 28 by 28. I mean, they're not super amazing   images and they're just black or white. So even if we're fantastic at this, they're  
never going to look great because we're using a small, simple data set. 
As you always should, when you're doing  any kind of R&D or experiments, you should 
always use a small and simple data set up  until you're so good at it that it's not   challenging anymore. 
And even then when you're exploring new ideas,  you should explore them on small, simple data  sets first. Yeah. 
So after I drew the various things, what I  like to do is one thing I found challenging 
about working with your class Tanishq is I  find when stuff is inside a class, it's harder 
for me to explore. So I copied and pasted it, the  
before_batch contents and called it noisify. And so one of the things that's fun to do that  
is it forces you to figure out what are the actual parameters to it.  And so now that I, rather than putting in the  class, now that I've got all of my, you know, 
various things to do with, so these are the  three parameters to the DDPM callbacks in 
it. And then these things we can calculate from that.  So with those then actually  all we need is yeah, what's the  
image that we're going to noisify and then what's the, what's the alphabar,   which I mean, we can get from here, but it's sort of be more general  
if you can pass in your alphabar. So yeah, this is just copying and pasting  
from the class, but the nice thing is then I could experiment with it.  So I can call noisify on my first 25 images  and with, with a random t, each one's got a 
different random t, and so I can print out  the t and then I could actually use those  as titles. And so this lets me, I thought this is quite nice. 
I might actually rerun this cause actually  none of these look like anything because as  it turns out in this particular  case, all of the t's are over 200. 
And as Tanishq mentioned, once you're over  200, it's almost impossible to see anything.  So let me just rerun this  and see if we get a better,  
there we go. There's a better one.  So with a t of 7, right? So remember t naught,  
t equals naught is the pure image. So t equals 7,   it's just a slightly speckledy image. And by 67, it's a pretty bad image. 
And by 94, it's very hard  to see what it is at all. 
And by 293, maybe I can see a pair of pants. I'm not sure I can see anything. 
So yeah. By the way,   there's a handy little, so we've, I think  we've looked at map before in in the 
course there's an extended  version of map in fastcore.  And one of the nice things is you can pass  it a string and it basically just calls this 
format string if you pass it a  string rather than a function.  And so this is going to stringify  everything using its representations. 
This is how I got the titles  out of it just by the way. 
So yeah, I found this useful to be  able to draw a picture of everything. 
And then I wanted to, yeah, look at what,  what, what else, what else can I do? 
So then I took nuts. You won't be surprised to see.  I took the sample method and  turn that into a function. 
And I actually decided to  pass everything that it needs.  Even, I mean, you could actually  calculate pretty much all of these. 
But I thought since I've calculated  them before, just pass them in.  So this is all copied and  pasted from Tanishq's version. 
And so that means the callback now is tiny, right?  Because before_batch is just noisify and the  sample method just calls the sample function. 
Now, what I did do is I decided just to, yeah,  I wanted to try like as many different ways 
of doing this as possible. Partly it's an exercise to help everybody like  
see all the different ways we can work with our framework, you know?  So I decided not to inherit from TrainCB,  but I instead I inherited from Callback. 
So that means I can't use Tanishq's  nifty trick of replacing predict. 
So instead I now need some way to pass in  the two parts of the first element of the 
tuple as separate things to the  model and return the sample. 
So how else could we do that? Well what we could do is we could actually   inherit from UNet2DModel, which is what Tanishq used directly, unit 2d  
model, and we could replace the model. And so we could replace specifically   the forward function. That's the thing that gets called. 
And we could just call the original forward  function, but rather than passing an x 
we’re passing a *x, and rather than  returning that, we'll return that .sample. 
Okay. So if we do that, then we don't need the   TrainCB anymore and we don't need the predict. And so if you're not working with something  
as beautifully flexible as miniai, you can always do this, you know, to make, to replace  
your model so that it has the interface that you need it to have. 
So now again, we did the same as  Tanishq had of create the callback.  And now when we create the model, we'll  reuse our UNet class, which we just created. 
I wanted to see if I can make things faster. I tried dividing all of Tanishq's channels  
by two and I found it worked just as well. One thing I noticed is that it uses group  
norm in the U-Net, which we have briefly learned about before and in group norm, it splits the  
channels up into a certain number of groups. And I needed to make sure that those  
groups had more than one thing in. So you can actually pass in how many groups  
do you want to use in the normalization. So that's what this is for. 
You gotta be a little bit careful of these  things, I didn't think of it at first and  I ended up, I think the num groups might've  been 32 and I got an error saying you can't 
split 16 things into 32 groups. But it also made me realize actually,   even in Tanishq's maybe you probably had 32 in the first with 32 groups. 
And so maybe the group norm  wouldn't have been working as well.  So they're little subtle things to look out for. So now that we're not using anything inherited  
from TrainCB, that means we either need to use TrainCB itself or just use our train learner  
and that everything else is the same as what Tanishq had.  So then I wanted to look at the results of  noisify here and we've seen this trick before, 
which is we call fit, but don't call the training  part of the fit and use the SingleBatchCB 
callback that we created way back  when we first created Learner.  And now learn.batch will contain the tuple  of tuples, which we can then use that trick 
to show. So I mean,   obviously we'd expect it to look  the same as before, but it's nice. 
I always like to draw pictures  of everything all along the way.  Cause it's very, very of.. I mean, I, the first six  
to seven times I do anything, I do it wrong. So given that I know that I might as well draw  
a picture to try and see how it's wrong until it's fixed.  It also tells me when it's not wrong. TANISHQ: Isn't there a show_batch function  
now that does something similar? JEREMY: Um, yes, you wrote that  
show_image_batch, didn't you? I can't quite remember.  Yeah. We should, uh, remind ourselves  
how that worked. That's a good point.  Thanks for a reminder. Okay. 
So then, um, I'll just go ahead and  do the same thing that Tanishq did.  And um, uh, but then the next thing I looked  at was I looked at the, you know, how am I 
going to make this train faster? I want a bigger, I want a higher learning rate.  Um, and I realized, oddly enough, the diffusers  code does not initialize anything at all. 
Make it faster: Initialization
They use the defaults, um, which just goes  to show like even, you know, the experts at 
Hugging Face that don't necessarily really  think like, oh, maybe the PyTorch defaults 
aren't, you know, perfect for my model. Of course they're not because they depend   on what activation function do you have and what res blocks do you have and so forth. 
Um, so I wasn't exactly sure how to initialize it. Um, I, um, partly by chatting to, um, Kat Crowson,  
who's the author of K-diffusion, um, and partly by looking at papers and partly  
by thinking about my own experience, I ended up doing a few things. 
One is I did do the thing that we talked about  a while ago, which is to take every second 
convolutional layer and zero it out. You could do the same thing with   using batch norm, which is what we tried. And since we've got quite a deep network,  
you know, that seemed like it might, you know, it helps basically by having the,   the, the non-id paths in the ResNets do nothing at first so they can't cause problems. 
Um, we haven't talked about, um, orthogonalized  weights before, and we probably won't because 
you would need to take our, um, computational  linear algebra course to learn about that, 
which is a great course, Rachel  Thomas did a fantastic job of it.  I highly recommend it, but I don't want to  make it a prerequisite, but, um, Kat mentioned, 
she thought that using orthogonal weights  for the downsamplers was a good idea. 
Um, and then, well, the up_blocks, they also  set the second convs to zero and something 
Kat mentioned, she found useful, which is  also from, um, I think it's from the Dhariwal 
Google paper is to also zero out the  weights of basically the very last layer. 
Um, and so it's going to start by predicting  zero as the noise, which is, you know, something 
that can't hurt. Um, so that was,   that's how I initialized the weights. Um, so call init_ddpm on my model, uh, something  
that I found that a huge difference is I replaced the normal Adam optimizer with one that has an   epsilon of 1e-5, the default, I think is 1e-8. 
And so to remind you, this is, we, is we,  when we divide by the exponentially weighted 
moving average of the squared gradients, we,  when we divide by that, if that's a very,  very small number, um, then it makes  the effective learning rate huge. 
And so we add this to it to make it not too huge. And it's nearly always a good idea to  
make this bigger than the default. I don't know why the default is so small.  And I found, until I did this, anytime I tried  to use a reasonably large learning rate somewhere 
around the middle of the 1-Cycle  training, it would explode.  Um, uh, so that makes a big difference. 
Um, so this way, yeah. Uh, I could train, I could get 0.016  
after 5 epochs. Um, and then  
sampling, so it looks all pretty similar. We've got some pretty nice textures, I think.  So then I was thinking, how do I get faster? So one way we can make it faster is we can take  
Make it faster: Mixed Precision
advantage of, um, something called mixed precision.  Um, so currently we're using  32 bit floating point values. 
Um, that's the defaults and  also known as single precision.  And um, GPUs are pretty fast at doing 32 bit  floating point values, but they're much, much, 
much, much faster during 16  bit floating point values.  So I'm 16 bit floating point values. 
I'm able to represent a very, you know, wide  range of numbers or much precision at the 
difference between numbers. And so they're quite difficult to use,   but if you can, you'll get a huge benefit because, um, modern GPUs, modern Nvidia GPU specifically  
have special units that do matrix multiplies of 16 bit values extremely quickly. 
Um, you can't just cast everything to 16 bit  because then you, there's not enough precision 
to calculate gradients and stuff properly.  So we have to use something  called Mixed Precision. 
Um, depending on how enthusiastic I'm feeling,  I guess we ought to do this from scratch as 
well. Um, we'll, we'll see.  Um, we do have an implementation from scratch  cause we actually implemented this before 
NVIDIA implemented it, um, in  an earlier version of fastai.  Um, um, anyway, we'll see. So basically the idea is that we use 32  
bit for things where we need 32 bit and we use Um, so that's what we're going to do is  
we're going to use this mixed precision. Um, but for now we're going to use, um, NVIDIA's,  
you know, semi-automatic or fairly automatic code to do that for us.  Actually we had a slight change of plan at  this point when we realized, uh, this lesson 
Change of plans: Mixed Precision goes to Lesson 20
was going to be over three hours in length  and we should actually split it into two.  So we're going to wrap up this lesson here and  we're going to, um, come back and implement 
this mixed precision thing in Lesson 20. So we'll see you then.

---

# 20

Hi and welcome to Lesson 20. In the last lesson  we were about to learn about implementing 
mixed precision training. Let's dive into it.  Um, so to, and I'm going to fiddle with other  things just because I want to really experiment. 
I just love fiddling around. So one thing I  wanted to do is I wanted to get rid of the 
DDPMCB entirely. We made it pretty small  here, but I wanted to remove it. So much as 
Tanishq said, isn't it great the callbacks  make everything so cool. I wanted to show  we can actually make things so cool without  callbacks at all. And so to do that, I realized 
what we could do is we could put noisify inside  a collation function. So the collation function, 
if you remember back to our datasets notebook,  which was going back to notebook five, and 
you've probably forgotten that by now. So  go and reread that to remind yourself it's 
the function that runs to take all of the,  you know, you've basically each kind of row 
of data, you know, will be a separate tuple,  but then it hits the collation function and  the collation function turns that into tensors,  you know, one tensor representing the independent 
variable, one tensor representing the dependent  variable, something like that. And the default 
collation function is called not surprisingly  default_collate. So if our collation function  calls that on our batch and then grabs the  X part, which is, it's always been the same 
for the last few things. That's the image cause  we use, cause data sets uses dictionaries.  So we're going to grab the image. Then we  can call noisify on that collated batch. Then 
that's exactly the same thing as Tanishq's  before_batch did, right? Because before_batch 
is operating on the thing that came out of  the default_collate function. So we could  just do it in a collate function. So if we  do it here and then we create a DDPM data 
loader function, which just creates a  DataLoader from some data set that we pass in  with some batch size, with that collation  function, then we can create our dls not using 
DataLoaders dot from what's it called?  DataLoaders.from_dd. But instead of that the, the, 
the original, you know, the plain init that  we created for DataLoaders, and again, you 
should go back and remind yourself of this.  You just pass in the data loaders for training  and test. So there's our two data loaders.  So with that, we don't need a DDPM callback 
anymore. All right. So now that we've, you know,  and again, this isn't, this is not required 
for mixed precision. This is just cause I  wanted to experiment and flex our muscles  a little bit of trying things out. So here's  our MixedPrecision callback, and this is 
MixedPrecision callback
a training callback. And basically if  you Google for PyTorch mixed precision,  
you'll see that the docs  
show the typical mixed precision  basically says with autocast device equals 
‘cuda’ type equals float16, get your predictions  and call your loss. So again, remind yourself 
if you've forgotten that this is called a  context manager and context managers, when  they start call something called dunder to  enter. And when they finish, they call something 
called dunder exit. So we could therefore  put the torch dot autocast into an attribute 
and calle dunder enter before the batch begins.  And then after we've calculated the loss, 
we want to finish that context manager. So  after_loss, we call autocast dunder exit. 
And so I had to add this. So  you'll find now in the 09_learner,  
that there's a section called updated version since the lesson   where I've added an after_predict and after_loss and after_backward and an after_step. And that  
means that a callback can now insert code at any point of the training loop.  
And so we haven't used all of those different things   here, but we certainly do want, yeah,  after_loss, we need to be able to do that. 
And then yeah, this is just code that has  to be run according to the PyTorch docs. So 
instead of calling loss dot backwards, you  have to call scaler.scale(loss).backward().  So we replace our backward in the train  callback. There's something called scaler.scale( 
loss).backward. And then it says that finally,  when you do the step, you don't call optimizer 
dot step. You call scaler.step(optimizer);  scaler.update(). So we've replaced step() 
with scaler.step(); scaler.update(). So  that does all of the things in here. And the 
nice thing is now that this exists, we don't  have to think about any of that. We can add 
mixed precision to anything which is really  nice. And so we now, as you'll see, cbs no 
longer has a DDPMCB. But we do have the  MixedPrecision and that's a train callback. So 
we just need a normal Learner, not a  TrainLearner. And we initialize our DDPM. Now to 
Getting the benefits from MixedPrecision
get benefit from mixed precision, you need  to do quite a bit at a time. You know, your 
GPU needs to be busy and on something as small  as Fashion-MNIST, it's not easy to keep a 
GPU busy. So that's why I've increased the  batch size by four times. Now that means that 
each epoch, it's going to have four times  less batches because they're bigger. And that 
means it's got four times less opportunities  to update. And that's going to be a problem  because if I want to have as good a result  as Tanishq had, and as I've had here in less 
time, that's the whole purpose of this, is  to do it in less time. Then I'm going to need 
to, you know, increase the learning rate and  maybe also increase the epoch. So increase 
the epochs up to 8 from 5 and I increase  the learning rate up to 1e-2. And 
yeah, I've found I could train it fine with  that once I used the proper initialization 
and most importantly use the optimization  function that has Epsilon of 1e-5. 
And so this trains, even though it's doing  more epochs, this trains about twice as fast 
and gets the same result.  
Does that make sense so far? TANISHQ: Yeah, it was great.  JEREMY: Cool. Now the good news is actually we  don't even need to write all this because there's 
HuggingFace Accelerator
a nice library from HuggingFace originally  created by Sylvain who used to work with me 
at fastai and went to HuggingFace and kept  on doing awesome work. And he started this  project called Accelerator, which he now works  on with another fastai alum named Zach Mueller. 
And accelerate is a library that provides this  single Accelerator that does things to accelerate 
your training loops. And one of the things  it does is mixed precision training. And it 
basically handles these things for you. It  also lets you train on multiple GPU's. It 
also lets you train on TPUs. So by adding a  TrainCB subclass that will allow us to use 
Accelerate, that means we can now hopefully  use TPUs and multi GPU training and all that 
kind of thing. So the Accelerate docs show  that what you have to do to use Accelerate 
is to create an Accelerator, tell it what  kind of mixed precision you want to use. So 
we're going to use 16 bit floating point fp16.  
And then you have to basically  call Accelerator.prepare 
and you pass in your model, your optimizer,  and your training and validation data loaders. 
And it returns you back a model and optimizer  and training and validation data loaders, 
that they've been wrapped up in Accelerate,  and Accelerate is going to now do all the 
things we saw you have to do, automatically.  And that's why that's almost all the code 
we need. The only other thing we need is, it  didn't, we didn't like tell it how to like 
change our loss function to use Accelerate.  So we actually have to change backward. That's  why we inherit from TrainCB. We have to  change backward to not call loss.backward,  
but self dot accelerate dot backward and pass in loss.  
Okay. And then I had another idea of  something I wanted to do, which is, 
I liked the idea that noisify, I've copied  noisify here, but rather than returning a  tuple of tuples, I just returned a tuple with  three things. I think this is neater to me. 
I would like to just have three things in  the tuple. I don't want to have to modify  my model. I don't want to have to modify my  training callback. I don't want to do anything 
tricky. I don't even want to have a custom  collation function‒ Sorry, I want to have 
a custom collation function, but I want to  have it‒ I don't want to have a modified model.  So I'm going to go back to using a UNet2DModel.  So how can we use a UNet2DModel when 
we've now got three things and  what I did in my modified Learner, 
just underneath it. Sorry, actually what I did  was I modified TrainCB to add one parameter, 
which is number of inputs. And so this tells  you how many inputs are there to the model. 
And normally you would expect one input, but  our model has two inputs. So here we say… 
Okay, so, AccelerateCB is a TrainCB. So  we, so when we call it, we say, we're going 
to have two inputs. And so what that's going  to do is it's just going to remember how many 
you asked for. And so when you call predict,  it's not going to pass learn.batch[0]. It's 
going to call *learn.batch[ :self.n_inputs].  And ditto when you call the loss function, 
it's going to be the rest. So   it's *learn.batch[self.n_inputs: ]  onwards. So this way you can have 
one, two, three, four, five inputs; one, two,  three, four, five outputs, whatever you like.  And it's just up to you then to make sure  that your model and your loss function take 
the number of parameters. So the loss function  is going to first of all, take your preds 
and then yeah, however many non-inputs you  have. So that way, yeah, we now don't need 
to replace anything except that we did need  to do the thing to make sure that we get the 
dot sample out. So I just had a little, this  is the whole DDPMCB callback now, DDPMCB2. 
So after the predictions are done, replace  them with the dot sample, right? So that's 
nice and easy, you know? So we ended up with  quite a bit of pieces, but they're all very 
decoupled, you know? So with miniai, you know,  with, and with AccelerateCB and whatever, 
because we should export these actually into  a nice module. Well, if you had all those, 
then yeah, you wouldn't have AccelerateCB.  The only thing you would need would be the,  would be the noisify and the collation function  and this tiny callback. And then yeah, here's 
our learner and fit and we get the same result  as usual. And this takes basically an identical 
amount of time because at this point I'm not  using multi GPU or TPU or whatever, I'm just  using mixed precision. So this is just a shortcut  for this. So it's not a huge shortcut. The 
main purpose of it really is to allow us to  use other types of accelerators or multiple  accelerators or whatever. So we'll look at  those later. Does that make sense so far? 
TANISHQ: Yeah. Oh yeah. Accelerate is  really powerful and pretty amazing. 
JEREMY: Yeah, it is. And I know like a lot of,  yeah, like I know Kat Crowson uses it in all her 
k-diffusion code, for example. Yeah. It's  used a lot out there in the real world. I've 
Sneaky trick: keep GPUs busy with MultDL
got one more thing I just want to mention  briefly, just this sneaky trick. I haven't 
even bothered training anything with it cause  it's just a sneaky trick. But sometimes thinking 
about speed, loading the data is the slow  bit. And so particularly if you use Kaggle, 
for example, on Kaggle, you get two  GPUs, which is amazing, but you only get  two CPUs, which is crazy. So it's really hard  to like take advantage of them because the 
amount of time it takes to like open a PNG or  a JPEG, you know, your GPU is sitting around 
waiting for you. So there's a, if your, you  know, if your data loading and transformation 
process is slow and it's difficult to keep  your GPUs busy, there's a trick you can do, 
which is you could create a new data loader  class, which wraps your existing data loader 
and replaces dunder iter. Now dunder iter  is the thing that gets called when you use a  for loop, right? Or when you use next data,  it calls this. And when you call this, you 
just go through the data loader as per usual.  So that's what dunder iter would normally do.  But then you also go through i from 0 to,  by default, 2, and then you spit out the 
batch. And what this is going to do is it's  going to go through the data loader and spit  out the batch twice. Why is that interesting?  Because it means every epoch is going to be 
twice as long. But it's going to only load  and augment the data as often as one epoch, 
but it's going to give you two epochs worth  of updates. And basically there's no reason 
to have a whole new batch every time. You  know, looking at the same batch two or three 
or four times, at a row, is totally fine. And  what happens in practice is you look at that 
batch, you do an update, get ready a part of  the weight space, look at exactly the same batch 
and find out now where to go in the, in, in  the weight space. It's still, yeah, basically 
equally useful. So I just wanted to add this  little sneaky trick here, particularly because 
if we start doing more stuff on Kaggle, we'll  probably want to surprise all the Kagglers 
with how fast our miniai solutions are. And  they'll be like, how is that possible? We'll 
be like, Oh, we're using our, you know, two  GPUs, extra Accelerate. I was thinking about  how do we use the two GPUs and like, Oh, and  we're, you know, using, you know, getting 
out, you know, loading, flying through using  MultDL. I think that'd be pretty sweet.  
So that's that.  JOHNO: Nice. TANISHQ:   Yeah. It's great to see the various different  ways that we can use miniai to, to do the 
Homework and experiment ideas
same thing, I guess. Um, or, you know, however  you feel like doing it or whatever works best 
for you. JEREMY:   Yeah. I'd be curious to see if other people  find other ways to, you know, I'm sure there's 
so many different ways to, to handle this  problem. I think it's an interesting, interesting 
problem to solve. And I think, for homework,  it'd be useful for people to, um, yeah. Run, 
run some of their own experiments, maybe either  use these techniques on other data sets or  see if you can come up with other variants  of these approaches, um, or come up with some 
different noise schedules, um, to try, um,  it would all be useful. Any other thoughts 
of exercises people could try? TANISHQ: Yeah. I mean, 
JOHNO: Getting away with  less than a thousand steps.  JEREMY: Yeah. Less than a thousand steps. JOHNO: Happening in the final 200. So  
why not just train with only 200 steps? JEREMY: Yeah. Less steps would be good.   Yeah. Because the sampling is  actually pretty slow. So that's 
a good point. TANISHQ: Yeah. Yeah.   Yeah. I was going to say something similar  in terms of like, yeah, many, I guess 
work with less number of steps, you know,  you would have to adjust the noise schedule  appropriately and you have to, I guess there's  maybe a little bit more thought into some 
of these things. Um, or, you know, another,  uh, aspect is like, um, when you're selecting 
the time step during training right now, we  selected randomly kind of uniformly each time 
step has equal probability of being selected.  Maybe different probabilities are better.  And some papers do analyze that more carefully.  So that's another thing to play around with 
as well. JEREMY: That's almost kind of like,  
if I guess there are almost two  ways of doing the same thing  in a sense, right? If you change that mapping  from T to beta, then you could reduce T and 
have different betas would kind of give you a  similar result as changing the probabilities 
of the T's, I think. TANISHQ: Yeah. I think,  
I think there's definitely, they're  both, they're kind of similar, but  potentially something complimentary happening  there as well. Uh, and I think those could 
be some interesting experiments to study that.  And also the, sort of, noise levels that you 
do choose affect the sort of behavior of, of  the sampling process. And of course, what, 
what features you focus on. Um, and so maybe  as, as people play around with that, maybe  they'll start to, to notice how using, yeah,  different noise levels or, you know, this 
different noise schedules affect maybe some  of the features that you see in the, in the  final image. Um, and that could be something  very interesting to the study as well. 
JEREMY: Great. Well, let me also say it's been  really fun. I'm doing something a bit different,  which is doing a lesson with you guys rather  than all on my lonesome. I, uh, I hope we 
can do this again because I've  really enjoyed it. TANISHQ: Yeah.   JEREMY: So of course now we're, you know, strictly speaking in the recording, we'll,  
we'll next up say Johno, who's actually already recorded his thanks to the Zoom mess up,  
but stick around. Uh, so I've already seen it. Johno’s thing is amazing. So you definitely  
don't want to miss that. JOHNO: Hello everyone.   Um, so today, uh, depending on the  order that this ends up happening, 
Style Transfer notebook
you've probably seen Tanishq’s DDPM implementation  where we taking the default training call 
back and doing some more interesting things  with preparing the data for the learner or  interpreting the results. Um, so in these  two notebooks that I'm going to show, we're 
going to be doing something similar, just  exploring like what else can we do besides  just the classic kind of classification model  where we have some inputs on a label and what 
else can we do with this miniai setup that we  have. And so in the first one, we can approach 
a kind of classic, um, AI art, uh, approach  called Style Transfer. And so the idea here 
is that we're going to want to somehow create  an artistic combination of two images where  we have the structure and, um, layout of one  image and the style of another. So we'll look 
at how we do that. And then we'll also talk  along the way in terms of like, why is this  actually useful beyond just making pretty  pictures. So to start with, I've got a couple 
of URLs for images. You're welcome to go and slip  in your own as well. And definitely recommend  trying this notebook with some different ones  just to see what effects you can get. Um, 
and we're going to download the image and,  um, load it up as a tensor. So we have here  a three channel image, 256 by 256 pixels.  And so this is the kind of base image that 
we're going to start working with. So before  we talk about styles or anything, um, let's  just think what is our goal here, right? We'd  like to, we'd like to do some sort of training 
or optimization. We'd like to get to a point  where we can match some aspect of this image. 
And so maybe a good place to start is to just  try and do, well, can we start from a random 
image and optimize it until it matches pixel  for pixel? Exactly. And that's going to help  us get this. Yeah. JEREMY: I think that  
might be helpful is if you type style  transfer deep learning into Google images, 
you could maybe show some examples so  that people will see what their goal. 
JOHNO: Yeah, that's a very good point. Um, so  let's see, this is a good one here. We've got the 
Mona Lisa as our, our base, but we've managed  to apply somehow some different artistic styles 
to that same base structure. So we have the Great  Wave by Kanagawa-oki, we have The Starry Night  by Vincent van Gogh, this is some sort of  Kandinsky or something. Um, yeah, so this 
is our end goal to be able to take the overall  structure and layout of one image and the  style from some different reference image. JEREMY: And in fact, this was the first, um,  
ever, I think fastai generative modeling lesson looked at style transfer. It's, um, it's been  
around for a few years. It's kind of a classic technique and it's really, um, I think a lot of   the students when we first did it, found it extremely useful,  
a way of better understanding, like,  you know, flexing their deep learning 
muscles, understanding what's going on and also  created some really interesting new approaches. 
So hopefully we'll see the same thing again.  Maybe some students will be able to show some  really interesting, um, um, results from this. JOHNO: Yeah. And I mean, today we're going to  
focus on kind of the classic approach. Um, but I  know one of the previous students from fastai  did a whole different way of doing that style 
loss and that we'll maybe post in the forums  or, you know, I've got some comparisons that  you can look at. Um, so yeah, definitely a  fruitful field still. And I think after the 
initial hype of like, everyone was excited  about style transfer apps and things, I don't  know, five years ago, um, I feel like there's  still some things to explore there. Very creative 
and fun little diversion  in the deep learning world.  Um, okay. So our first step in getting to  that point is being able to optimize an image. 
Optimizing an image
And so up until now we've been optimizing  like the weights of a neural network. Um,  but now you want to go to something a bit  more simple and we just want to optimize the 
raw pixels of an image. JEREMY: Do you mind if we scroll   up a bit to the previous code just  so we can have a look at it? So 
there's a couple of interesting points about  this code here is, you know, we're not, we're  not cheating. Um, well not really. So we're,  so yeah, we, we've seen how to download things 
in the network before. So we're using  fastcore urlread, cause we're allowed to. And  then I think we decided we weren't going to  write our own JPEG parser. So torchvision 
actually has a pretty good one, which a lot  of people don't realize exists. And a lot  of people tend to use PIL, but actually  torchvision has a more performant option. Um, and 
it's actually quite difficult to find any  examples of how to use it like this. Um, but 
here's some code you can borrow. JOHNO: Yeah. And if you Google “load image   from url in pytorch”, all of the examples are going to use PIL. And that's what I've done  
historically, um, is use the requests library to download the URL and then feed that into  
PIL’s Image.open function. Um, so yeah, that was fun when   I was working with Jeremy on this  notebook, like that's how I was doing 
it. It's wait, we're breaking the rules. Uh,  let's see if we can do this directly into  a tensor without this intermediate step of  loading it with, with Pillow. Um, cool. Okay. 
So how are we going to do this image optimization?  Um, well, first thing is we don't really have 
a dataset of lots of training examples. We  just have a single like target and a single  thing we're optimizing. Um, and so we've built  this LengthDataset here, which is just going 
to follow the PyTorch like dataset standard.  We can tell it how to get a particular item  and what our length is. Um, but in this  case, we're just always going to return 0, 0. 
and we're not actually going to care about  the results from this dataset. We just want  something that we can pass to the learner  to do some number of training iterations. 
So we create like a fake dummy dataset with  a hundred items. Um, and then we create a 
data loaders from that, and that's going to  give us a way to train for some number of  steps without really caring about what this  data is. Um, so does baking make sense? 
JEREMY: Yeah. So just to clarify the reason  we're doing this. So basically the idea is we're  going to, um, start with  that photo you downloaded.  
Um, and I guess you're going to be downloading another photo. So that photo is going to be   like the content. We're going to try to make it continue to look like that lady. And then  
we're going to try to change the style so that the style looks like the style of some   other picture. And the way we're going to be doing that is by doing an optimization  
loop with like SGD or whatever. But so the idea is that each step of that,  
we're going to be moving the style somehow of the image closer and closer to one of those images  
you downloaded. So it's not that we're going to be looping through lots of different images,   but we're just going to be looping through steps of an optimization  
loop. Is that the idea? JOHNO: Exactly.   Um, and so yeah, we can, we can create  this data loader. And then in terms of the 
actual like model that we're optimizing and  passing to the learner, and we've created  this tensor model class, which just has whatever  tensor we pass in as its parameter. So there's 
no actual neural network necessarily. We're  just going to pass in a random image or, or  some image shaped thing, a set of  numbers that we can then optimize. 
JEREMY: So just in case people have forgotten  that. So to remind people, when you put something  in an end up parameter, it doesn't change  it in any way. It's just a normal tensor, 
but it's stored inside the module as being  something as being a tensor to optimize. So 
what you're doing here, Johno, I guess, is  to say, I'm not actually optimizing a model  at all. I'm optimizing an image,  the pixels of an image directly. 
JOHNO: Exactly. Um, and because it's in a  parameter, if we look at our model, we can 
see that, for example, model.t… it does  require_grad, right? Because that's already 
set up because this nn.Module is going to  look for any parameters. And if our optimizer 
is looking at, um, let's look  at the shape of the parameters.  
Um, so this is the shape of the parameters that we're optimizing.   This is just that tensor that we passed in, the same shape as our image. And this is what's  
going to be optimized if we pass this into any sort of learner fit method. JEREMY: Okay.   So this model does have a thing being passed to forward, which is x, which we're  
ignoring. And I guess that's just because our learner passes something in. So we're   making life a bit easier for ourselves by making the model look the way our learner  
expects. JOHNO: Yeah. JEREMY: And  we could have used anything like  TrainCB or something if we wanted to, but this  seems like a nice, nice and easy way to do 
it. JOHNO: Yeah. So, I mean,  this is the way I've done  it. Um, if you do want to use TrainCB, you can  set it up, um, with a custom predict method 
that is just going to call the model forward  method with no parameters. Um, and if you want 
likewise, just calling the loss function on  just the predictions. Um, but if you want 
to skip this, because we take this argument x  equals zero and never use it, um, that should  also work without this callback. So either  way is fine. Um, this is a nice approach if 
you have something that you're using an existing  model, which expects some number of parameters  or something. Um, yeah, you can just modify  that training callback, but we almost don't 
Loss function and Learner
need to in this case. Um, okay. So let's see,  let's put this in a learner. Let's optimize 
it with some loss function. JEREMY: Oh,  just to clarify, I get it. So the loss,  the get_loss you had to change  because normally we pass a  
target to the loss function. JOHNO: Yeah. We, we, yeah,   this is learn.preds. I mean learn.batch...  JEREMY: And again, we could, we could avoid, 
we could remove that as well if we wanted to,  by having our loss function, take a target  that we then ignore. JOHNO: Yeah, yeah,  exactly. JEREMY: Cool. JOHNO: Um, so both, both 
other approaches, I like this because it's,  we can kind of be building on this idea of  modifying the training call back in the DDPM  example and the other examples. Um, but in 
this case, it's just these like two lines  change. This is how we get a model predictions.  We just call the forward method, which returns  this image that we're optimizing and we're 
going to evaluate this according to some loss  function that just takes in an image. And 
so for our first loss function, we're just  going to use the mean squared error between  the image that we are generating, like this  output of our model and that content image. 
That's our like target, right? So we're going  to set up our model, start it out with a random  image like this above. We're going to create  a learner, um, with a dummy data loader for 
a hundred steps. Our loss function is going  to be this means squared error loss function,  a set of learning rates and an optimizer  function. The default would probably also 
work. Um, and if we run this, something's  going to happen, our loss is going to go from  a non-zero number to close to zero. Um, and  we can look at the final result. Like if we 
call learn.model and show that as an image  versus the actual image, we'll see that  they look pretty much identical. JEREMY: Yeah.  So just to clarify, this was like a, a pointless 
example, but what we did, we started with  that noisy image you showed above, and then 
we used SGD to make those  pixels get closer and closer to   the lady in the sunglasses. Not, you know, not for any particular purpose,  
but just to show that we can turn noisy pixels into something else by having it follow  
a loss function. And this loss function was just like: make the pixels look as much as   possible, like the lady in the sunglasses. JOHNO: Exactly. And so in this case, it's a  
very simple loss. There's like a one direction that you update. So it's, it's almost trivial   to solve, but it still helps us get like the framework in place. Um, but just seeing this  
final result is not very instructive because you almost think, well, did I get a bug in   my code that I just duplicated the image? How do I know this is actually doing what we  
Viewing progress: ImageLogCB
expect? And so before we even move on to any more complicated loss functions,   um, I thought it was important to have some sort of more obvious way of doing progress. Um, so I've  
created a little, um, logging callback here that is just after every batch,  
um, it's going to store the output as an image. And then… JEREMY: I guess after  
every 10 batches here by default. JOHNO: Oh yes. Yeah. Sorry. So we can set   how often it's going to update and then every 10 iterations or 50 iterations, whatever we  
set the log, every argument to, it's going to store that in the list. Um, and then after  
the training is done after fits, we're just going to show those images.   Um, and so everything else, the  same as before, the passing in this 
extra logging callback, it's going to give  us the kind of progress. And so now you can 
see, okay, there is actually something happening.  We starting from this noise after a few iterations  already most of it is gone. And by the end  of this process, it looks exactly like the 
content image. JEREMY: So I really like this  because what you've basically done here is you've 
now already got all the calling and infrastructure  in place you need to, basically, create a really 
wide variety of interesting outputs that could  either be artistic or, um, or like, you know, 
they could be more like imagery construction,  super resolution, colorization, whatever. 
And you just have to modify the loss function  and you, you know, and I really like the way 
you've created the absolute easiest possible  first and fully checked it. And before you 
start doing the fancy stuff and now you kind  of, I guess you're really comfortable doing 
the fancy stuff because you  know, that's all in place.  JOHNO: Yeah, exactly. And we know that we're  going to see some tracking. So hopefully it'll be 
like visually obvious if things are going  wrong and we know exactly what we need to  modify. If we can now express some desired  property, it's more interesting than just, 
like, mean squared error to a target image.  Then we quickly have everything in place to  optimize. And so this is not really fun to  like, okay, let's think about what other loss 
functions we could do. Maybe we want it to  match the image, but also have a particular  overall color. Maybe we want some, some more  complicated thing. And so towards that, like 
towards starting to get a more richer, like  measure of what this output image looks like. 
Extracting features from a pre-trained network, VGG16
We're going to talk about extracting features  from a pre-trained network. And this is kind  of like the core idea of this notebook is  that we have these big convolutional neural 
networks. This one is a much older  architecture and so relatively   simple compared to some of the big, you know,  
DenseNets and so on used today. JEREMY: It's actually a lot like our   pre ResNet Fashion-MNIST model  is basically almost the same as 
VGG16. JOHNO: Yeah, yeah, exactly. And so we feeding in  
an image and then we have these like convolutional layers, downsampling, convolution, you know,   downsampling with max pooling up until some final prediction. 
JEREMY: Oh, can I just point something out?  There's one big difference here, which is that  7 by 7 by 512, if you can point at that.  Normally nowadays and in our models, 
we tried, you know, using an adaptive or  global pooling to get down to a 1 by 1 by 512, 
VGG16 does something which is very unusual by  today's standards, which is it just flattens 
that out into a 1 by 1 by 4,096. Which  actually might be a really interesting 
feature of VGG and I've always felt like people  might want to consider training, you know, 
ResNets and stuff without the global  pooling and instead do the flattening.  The reason we don't do the flattening nowadays  is that that very last linear layer that goes 
to 1 by 1 by 4,096 to 1 by 1 by  1,000 ‒because this is an ImageNet  model‒ is going to need an awfully big  weight matrix. You've got a 4,096 x 1,000 
weight matrix as a result  of which this is actually   horrifically memory intensive for a reasonably poor performing model by  
modern standards. But yeah, I think that doing that actually also has some, some benefits  
potentially as well. JOHNO:   Yeah. And in this case, we are not even really  interested in the classification side. We're 
more excited about the capacity of this to  extract different features. And so, the idea 
here, and maybe I should pull up this classic  article looking at like, what do neural networks 
learn and trying to visualize some of these  features. This is something we've mentioned  before with these big pre-trained networks  is that the early layers tend to pick up on 
very simple features, edges and shapes and  textures. And those get mixed together into  more complicated textures. And by the way,  this is just trying to visualize what kind 
of input maximally activates a particular  output on each of these layers. And so it's 
a great way to see what kinds of things that's  learning. And so you can see as we move deeper  and deeper into the network, we're getting more  and more complicated like hierarchical features. 
JEREMY: We should mention, so we've looked at  the Zeiler and Fergus paper before that, which  is an earlier version doing something like this  to see what kind of features were available. 
So we're linked to this distill paper from  the forum and the course lesson page, because 
it's actually a more modern and fancy  version kind of, of the same thing.  JOHNO: Yeah. Also note the  names here. All of these  
people are worth following. Chris does amazing work on interpretability and Alexander Mordvintsev   we'll see in the second notebook that I look at today and doing all sorts of other cool  
stuff as well. Anyway, so we want to think about like, let's extract the outputs of  
these layers in the hope that  they give us a representation  of our image that's richer than just  the raw pixels. So we can list... 
JEREMY: The idea being there  that if we had another,   if we were able to change our image to have the same features at those various types that  
you were just showing us, that then it would like have similar textures or similar  
kind of higher level concepts or whatever. JOHNO: Exactly. So if you think of this like 14   by 14 feature map over here, maybe it's capturing that there's an eye in the top left and some hair  
on the top right, these kind of abstract things. And if you change the brightness  
of the image, it's unlikely that it's going to change what features are stored there because   the networks land to be somewhat invariant to these like rough transformations, a bit  
of noise, a bit of changing texture early on is not going to affect the fact that   it still thinks this looks like a dog and a few days before that, that it still thinks that part  
looks like a nose and that part looks like an ear.  JEREMY: Maybe the more interesting bits then for  what you're doing are those earlier layers where 
it's going to be like, there's a whole bunch  of kind of diagonal lines here, or there's  a kind of a loopy bit here. Because then yeah,  if you replicate those, you're going to get 
similar textures without changing the semantics. JOHNO: Exactly. Yeah. So I mean, I guess let's  
load the model and look at what the layers are.  And then in the next section we can try and  like see what kinds of images work when we 
optimize towards different layers in there.  So this is the network we have, Convolutions, 
ReLUs, Max Pooling, all of this we should  be familiar with by now. And it's all just 
in one big nn.Sequential. This doesn't have  the head. So we said dot features. If you 
did this without, you'd have then the, this is  like the features sub network. That's everything 
up until some point. And then you have the  flattening and the classification, which we  are kind of just throwing away. So this is  the body of the network. And we're going to 
Normalizing the image
try and tag into various layers here and extract  the outputs. But before we do that, there's 
one more bit of admin we need to handle. This  was trained on a normalized version of ImageNet, 
right? Where you took the dataset mean and  the dataset standard deviation and use that to  normalize your images. So if we want to match  what the data looked like during training, 
we need to match that normalization step.  And we've done this on grayscale images where  we just subtract the mean, divide by the standard  deviation. But with three channel images, 
these RGB images, we can't get away with just  saying, let's subtract our mean from our image 
and divide by the standard deviation. You're  going to get an error that's going to pop  up. And this is because we now need to think  about broadcasting and these shapes a little 
bit more carefully than we can with just a  scale of value. So if we look at the mean  here, we just have three values, right? One  for each channel, the red, green, and blue 
channels. Whereas our content image has 3  channels and then 256 by 256 for the spatial 
dimensions. So if we try and say content image  divided by the mean or minus the mean, it's 
going to go from right to left and find the  first non-unit access. So anything with a 
size greater than one, and it's going to try  and line those up. And in this case, the 3  and the 256, those are going to match. And so  we're going to get an error. More perniciously, 
if the shape did happen to match, that might  still not be what you intended. So what we'd  like is to have these 3 channels mapped to  the three channels of our image, and then 
somehow expand those values out across the  2 other dimensions. And the way we do that 
is we just add two additional dimensions on  the right for our imagenet_mean. And you could 
also do the unsqueeze minus one, the unsqueeze  minus one. But this is the kind of syntax 
that we're using in this course. And now our  shapes are going to match because we're going  to go from right to left. If  it's a unit dimension, size one,  
we're going to expand it out to match the other tensors.   And if it's a non-unit dimension,  then the shapes have to match. 
And that looks like it's the case. And so  now with this reshaping operation, we can 
write a little normalize function, which we  can then apply to our content image. And I'm  just checking the min and the max to make  sure that this roughly makes sense. We could 
check the mean as well to make sure that the  mean is somewhat close to zero. Okay, in this 
case less maybe because it's a darker image  than average. But at least we are doing the  operation that seems like the math is  correct. And now the shape is not the same. 
JEREMY: Maybe doing the channel-wise  mean would be interesting.  JOHNO: Oh, yes. So that would be the mean  over the dimensions. 1 and 2, I think.  
I think you have to tap all 1 comma 2.  Yes. I wasn't sure which way around it was. Yeah, I always forget to. 
Okay, so our blue channel is brighter than  the others. And if you go back and look at  our image, you could maybe believe that the  image int is going to be blue and red and 
the face is going to be just blue. Yeah, okay.  So that seems to be working. We can double 
check because now that  we've implemented ourselves,   torchvision.transforms has a normalized function that you can pass the mean and standard deviation  
to, and it's going to handle making sure that the devices match, that the shapes match,   et cetera. And you can see if we check the min and max, it's exactly the same, just a little  
bit of reassurance that our function is doing the same thing as   this normalized transform. JEREMY: I appreciate you not  
cheating by implementing that. Johno, thank you. JOHNO: You're welcome. Got to follow the rules.  JEREMY: Got to follow the rules. JOHNO: Okay. So a bit of admin out the way.  
Intermediate representations, features
We can finally say how do we extract the features from this network. And now if you remember the  
previous lesson on hooks, that might be something that springs to mind. I'm going   to leave that as an exercise for the reader. And what we're going to do is we're just going  
to normalize our input and then we're going to run through the layers one by one in  
this sequential stack. We're  going to pass our x through  that layer. And then if we're in one of the  target layers, which we can specify, we're 
going to store the opposite of that layer. JEREMY: And I can't remember if I've used the   term “features” before or not, so apologies if I have, but just to clarify here, when we say  
features, we just mean the activations of a layer. And in this case, Jono's picked out  
two particular layers, 18 and 25. I just want to, I mean, I'm not sure it matters in this  
particular case, but there's a bit of a gotcha you've got here, Johno, which is you should   change that default 18, 25 from a list to a tuple. And the reason for that is that when  
you use a mutable type, like a list in a Python default parameter, it does this really weird  
thing where it actually keeps it around. And if you change it at all later than it actually   kind of modifies your function. So I would suggest, yeah, never using a list as a default  
parameter because at some point it will create the weirdest bug you've ever had. I speak,  
I say this in experience. JOHNO: Yeah, yeah. That sounds like something   that was hard won. All right. I'll change that. And by the time you see this notebook,  
that change should be there. All right. So this is one way to do it. Just manually running through   the layers one by one up until whatever the latest layer we're interested in is.  
But you could do this just as easily by adding hooks to the specific layers and then just feeding   your data through the whole network at once and relying on the hooks to  
store those intermediates. JEREMY: Yeah. So let's make   that homework actually, not just  an exercise you can do, but yeah, 
Hooks homework
I want, let's make sure everybody does that.  You can use the, one of the hooks callbacks 
we had or the hooks context managers we had, or  you can use the register forward hook PyTorch 
directly. JOHNO:   Yeah. And so what we get out here, we feeding  in an image that's 256 by 256. And the first 
layer that we're looking at is this one here.  And so it's getting half to 128, then to 64. 
These ones are just different because it's  a different starting size. And then to 32  by 32 by 512. And so those are the features  that we're talking about for that layer 18. 
It's this thing of shape 512 by 32 by 32 for  every kind of spatial location in that 32 
by 32 grid, we have the output from 512 different  filters. And so those are going to be the 
features that we're talking about. JEREMY: So there's a [unintelligible]   being the channels in a single convolution. JOHNO: Yeah. Okay. So what's the point of  
Optimizing an image with Content Loss
this? Well, like I said, we’re  hoping that we can capture  different things at different layers. And so  to kind of first get a feel for this, like, 
what if we just compared these feature maps  we can institute what I'm calling a content  loss, or you might see it as a perceptual  loss. And we're going to focus on a couple 
of later layers. Again, make sure that this  is... as I've learned. And what we're going to 
do is we're going to pass in a target image,  in this case, our content image. And we're  going to calculate those features in  those target layers. And then in the  
in the forward method, when we're comparing to our inputs,   we're going to calculate the features of our inputs. And we're going to do the mean squared  
error between those and our target features. So maybe there's a bad way   of explaining it. But the so the… JEREMY: I can maybe read it back to you to make  
sure you understand. JOHNO: Yeah. So good idea. JEREMY: Okay. So this is a loss function you've  
created. It has it done to  call method, which means  you can pretend that it's a function. It's  a callable in Python language. Your forward… 
your best… So yeah, in nn.Module would call  it forward. But in normal Python, we just 
use it dunder call. It's taking one input,  which is the way you set up your image training 
callback earlier, it's just going to pass  in the input, which is this is the image as  it's been optimized to so far. So initially,  it's going to be that random noise. And then 
the loss you're calculating is the mean squared  error of how far away is this input image 
from the target image, the mean squared error  for each of the layers, by default 18 and 
25. And so you're literally actually, it's a  bit weird, you're actually calling a different 
neural network; calc_features actually call  the neural network, but not because that's 
the model we're optimizing, but because it's  actually, the loss function is how far away 
are we. Yeah. So that's the loss function. And  so if we, so if you're, so if we, with SGD, 
optimize that loss function, you're not going  to get the same pixels. You're going to get, 
I don't even know what this is going to look  like. You're going to get some pixels, which  have the same activations  of those features. JOHNO:  
Yeah. And so if we run that, we see, you can see the sort of shape of our person there,  
but it definitely doesn't match on like a color and style basis. JEREMY: So 18 and 25 remind  
us how deep they are in the scheme of things. JOHNO: So these are fairly close towards  
the end. JEREMY: Okay. So I  guess color often doesn't have  much of a semantic kind of property. So that's  probably why it doesn't care much about color 
because it's still going to be an eyeball,  whether it's green or blue or brown. JOHNO: Yeah. 
There's something else I should mention, which  is we aren't constraining our tensor that 
we're optimizing to be in the same bounds  as a normal image. And so some of these will  also be less than zero or greater than one  as kind of like almost hacking the neural 
network to get the same features that those  deep layers by passing in something that it's  never seen during training. And so for display,  we're clipping it to the same bounds as an 
image, but you might want to have either some  sort of sigmoid function or some other way  that you clamp your tensor model and to have  outputs that are like within the allowed range 
for. JEREMY: Oh, good point.  Also, it's interesting to  note the background hasn't changed much. And  I guess the reason for that would be that 
the VGG model you were using in  the lost function was trained on   ImageNet and ImageNet is specifically about recognizing generally a single  
big object, like a dog or a boat or whatever. So it's not going to care about the background  
and the background probably isn't going to have much in the way of features at all, which   is why it hasn't really changed the background. JOHNO: Yeah, exactly. And so, I mean, this is  
kind of interesting to see how little it looks like the image will at the same time still being   like, if you squint, you can recognize it. But we can also try passing in  
earlier layers, right? And  comparing on those earlier  layers and see that we get a completely different  result because now we're optimizing to some 
image that is a lot closer to the original.  It still doesn't look exactly the same. And 
so there's a few things that I thought were  worth noting, just potentially of interest.  One is that we're looking into these ReLUlayers,  which might mean, for example, that if you're 
looking at the very early layers, you're missing  out on some kinds of features. That was one  of my guesses as to why this didn't have as  dark a darks as the input image. And then 
also we still have this thing where we might  be going out of bounds to get the same kinds  of features. So yeah, you can see how by looking  at really deep layers, we really don't care 
about the color or texture at all. We're just  getting like sunglassesy bits and nosey bits  there. By looking at the earlier layers, we  have much more rigid adherence to the sort 
of lower level features as well. And so this  is nice. It gives you a very tunable way to  compare two images. You can say, do I care  that they match exactly on pixels? Then I 
could use mean square error. But do I care  quite a lot about the exact match? Then I  can use maybe some early layers. But do I  only care about the overall semantics? In 
that case, I can go to some deeper  layers and you can experiment with.  JEREMY: If I remember correctly,  this is also something like the  
kind of technique that Zeiler and Fergus and the Distill.pub papers use to like   just identify like what do filters look at, which is like, you can optimize an image  
to try and maximize a particular filter, for example, that would be a similar loss function   to the one you've built here. And that would show you, yeah, what they're,  
what they're looking at. JOHNO: Yeah. And that would be a really fun little   project actually. So do it where you calculate these feature maps and then just pick one of  
those 512 features and optimize the image to maximize that activation. By default,  
you might get quite a noisy, weird result, like almost an adversarial input. And so what these   feature visualization people do is they add things like augmentations so that you're  
optimizing an image that  even under some augmentations  still activates that feature. But, yeah,  that might be a good one to play with. 
Cool. Okay. So we have a lot of our infrastructure  in place. We know how to optimize an image.  We know how to extract features from this  neural network and we're saying this is great 
for comparing at these different kind of types  of feature how similar two images are. The 
final piece that we need for our full style  transfer, artistic application is to say I'd 
like to keep the structure of this image,  but I'd like to have the style come from a  different image. And you might think, oh,  well, that's easy. We just look at the early 
layers like you've shown us. But there's a  problem, which is that these feature maps  by default, we feed in our image and we get these  feature maps. They have a spatial component, 
right? We said we had a 32 by 32 by 512 feature  map out and each of those locations in that 
32 by 32 grid are going to correspond to some  part of the input image. And so if we just 
said, let's do mean squared error for the  activations from some early layers, what we'd  be saying is I want the same types of feature,  like the same style, the same textures, and 
I want them in the same location. Right? And so  we can't just get like Van Gogh brushstrokes. 
We're going to try and have the same colors  in the same place and the same textures in  the same place. And so we're going to get  something that just matches our image. What 
we'd like is something that has the same colors  and textures, but doesn't, but they might  be in different parts of the image. So we  want to get rid of this spatial aspect. 
JEREMY: So just to clarify, when we're saying to  it, for example, give it to us in the style of 
Van Gogh's Starry Night, we're not saying  in this part of the image, there should be 
something with this texture, but we're saying  that the kinds of textures that are used anywhere  in that image should also appear in our version,  
but not necessarily in the same place. JOHNO: Exactly. And so the solution  
Style Loss with Gram Matrix
that Zeiler and Fergus proposed is  this thing called a Gram Matrix.  So what we want is some measure of what kinds  of styles are present without worrying about 
where they are. And so there's always a trouble  trying to represent more than two dimensional 
things on a 2D grid. But what I've done here  is I've made our feature map, right, where  we have our height and our width, that might  be 32 by 32 and some number of features, but 
instead of having those be like a third dimension,  I've just represented those features as these  little colored dots. And so what we're going  to do with a Gram Matrix is we're going to 
flatten out our spatial dimension. So we're  going to reshape this so that we have the 
width times the height so that, like, the spatial  location on one axis and the feature dimension  on the other. So each of these rows is like  this is the location here. There's no yellow 
dots, so we get a zero. There's no green,  so we get a zero. There is a red and a 
blue, so we get ones. So we've kind of flattened  out this feature map into a 2D thing. And 
then instead of caring about the spatial dimension  at all, all we care about is which features 
do we have in general, which types of features,  and do they occur with each other. And so 
we're going to get effectively the dot products  of this row with itself and then this row 
with the next row and this row with the next  row. We're saying like for these feature vectors, 
how correlated are they with each other?  Right. And so we'll see this in code just now. 
JEREMY: I think you might've  said, I might've misheard you,  
but I just want to make sure I got the citation here, right? So this idea came from,  
I don't know if it was first invented in the Gatys et al. paper, called “A Neural  
Algorithm of Artistic Style”. JOHNO: Yeah. Yeah. I meant Gatys,  
that's the style transfer one.  Zeiler and Ferguson is the feature  visualization one. Yeah. Sorry, I got those  switched. Thanks, Jeremy. Okay. So we are 
ending up with this kind of like, this Gram  matrix, this correlation of features. And  the way you can read this in this example is  to say, okay, there are seven reds, right? 
Red with red, there's seven in total. And if  you go and count them, there's seven there. 
And then if I look at any other one in this  row, like here, there's only one red that  occurs alongside a green, right? This is the  only location where there's one red cell and 
one green cell. There's three reds that occur  with the yellow. They're there and there.  And so this gram matrix here has no spatial  component at all. It's just the feature dimension 
by the feature dimension. But it has a measure  of how common these features are. Like what's 
an uncommon one here? Yeah, maybe there's  only three greens in total, right? And all 
of them occur alongside a yellow. One of them  occurs alongside a red. One of them occurs  alongside a blue. Yeah, so this is exactly  what we want. This is some measure of what 
features are present, where if they occur  together with other features often, that's 
a useful thing, but it doesn't have the  spatial component. We've gotten rid of that.  JEREMY: And this is the first clear explanation  I've ever seen of how a Grain Matrix works. This 
“A Neural Algorithm of Artistic Style” paper
is such a cool picture. I also want to, maybe  you can open up the original paper because 
I'd also like to encourage people to look at  the original paper because this is something  we're trying to practice at this point is  reading papers. And so hopefully you can take 
Johno's fantastic explanation  and yeah, and bring it back to  
understanding the paper as well.  
That's crazy that it's been so far down. JOHNO:  Oh yeah, it's a different search engine that 
I'm trying out that has some AI magic, but  they use Bing for their actual searching.  [...] Yeah. So we can quickly  check the paper. I don't know 
if I've actually read this paper ‒as horrific as  that sounds. JEREMY: Not horrific at all. It was 
a while ago. But I think it's  got some nice pictures and  
I'm going to zoom in a bit. JOHNO: Oh, good idea.  
Okay. They're the examples. JEREMY: It's  great examples. Yeah. Love for Kandinsky.  JOHNO: Sorry about the doorbell. Okay. Yeah. Gram  Matrix “inner a product between the vectorized 
feature map”. So those kinds of wordings kind of  put me off on for a while. The way I explained 
Gram Matrices when I had to deal with them  at all was to say it's magic that measures  with, you know, what features are there without  worrying about where they are and left it 
at that. But it is worth trying to decode this  back. They talk about which layers they're 
looking into. I think in TensorFlow they have  names. We're just using the index. Okay. Yeah. 
So it doesn't really explain how the Gram  Matrix works, but it's something that people  use historically in some other contexts  as well. And for the same kind of measure. 
JEREMY: Nowadays actually PyTorch has  named parameters and I don't know if  
they've updated VGG yet, but you can name   layers of a sequential model as well. JOHNO: Yeah. Okay. So just quickly,  
I wanted to implement this diagram  in code. I should mention these  are, these are like zero or one for simplicity,  but you could have like obviously different 
size activations and things. The correlation  idea is still going to be there. Just not  as easy to represent visually. And so we're  going to do it with an einsum because it 
makes it easy to add later the bash dimension  and so on. But I wanted to also highlight 
that this is just this matrix multiplied with  its own transpose and you're going to get 
the same result. So yeah, that's our Gram  Matrix calculation. There's no magic involved 
there as much as it might seem like it. And  so we can now use this, like, can we create 
this measure and then can we... JEREMY: When you look later at   things like Word2Vec, I think it's  got some similarities, this idea 
of kind of co-occurrence of features. And  it also reminds me of the clip loss similar 
idea of like basically a dot product, but  in this case with itself, I mean, we've seen 
how covariance is basically that as well. So  this idea of kind of like multiplying with 
your own transpose is a really common mathematical  technique we've come across three or four 
times already in this course. JOHNO: Yeah. And it comes up all   over the place even. Yeah. You'll  see that in protein folding stuff 
as well. They have a big  covariance matrix for like...  JEREMY: So the difference in each case is like,  yeah, the difference in each case is the matrix 
that we're multiplying by its own transpose.  So for covariance, the matrix is the matrix  of differences to the mean, for example. And  yeah, in this case, the matrix is this flattened 
picture thing. JOHNO: Cool. So I have here  
the calc_grams function that's  going to do exactly that operation  we did above, but we're going to add some  scaling. And the reason we're adding the scaling 
is that we have this feature map and we might  pass in images of different sizes. And so  what this gives us is the absolute... You  can see there's a relation to the number of 
spatial locations here. And so by scaling  by this width times height, we're going to 
get a relative measure as opposed to an absolute  measure. It just means that the comparisons  are going to be valid even for images of different  sizes. And so that's the only extra complexity 
here, but we have channels by height, by width,  image in, and we're going to pass in... Oh, 
sorry. This is channels being the number of  features. We're going to pass it in two versions 
of that, right? Because it's the same image  in both times. But we're going to map this 
down to just this features by features,  but you can't repeat variables and einsum.  So that's why it's c and d. And if we run this  on our style image, you can see I'm targeting 
five different layers. And for each one, the  first layer has 64 features. And so we get 
a 64 by 64 Gram Matrix. The second one has 128  features. We can get 128 by 128 Gram Matrix. 
So this is doing, it seems like what we want.  Because this is a list, we can use this attrgot 
method which I... JEREMY: Well,   actually it's a fastcore capital L, not a list. JOHNO: Oh, sorry. Yeah. Magic list. And so I  
like to think. Magic list.  Yeah. So either works. Okay. So let's use  this as a loss. Just like with the content 
loss before, we're going to take in a target  image, which is going to be our style. We're  going to calculate these gram matrices for  that. And then when we get in an input to 
our loss function, we're going to calculate  the gram matrices for that and do the mean  squared error between the gram matrices. So  these are the no spatial components, just 
what features are there, comparing the two  to make sure that they ideally have the same  kinds of features and the same kinds  of correlations between features.  
So we can set that up. We can evaluate it on my image. So our   content image at the moment has quite a high loss when we compare it to our style image. 
JEREMY: That means that the content image doesn't  look anything like a spider web in terms of  its textures and whatever. JOHNO:  
Optimizing to get the final result
Exactly. So we're going to set up an optimization  thing here. One difference is that at the  moment I'm starting from the content image  itself rather than optimizing from random 
noise. You can choose either way. For style  transfer, it's quite nice to use the content 
image as the starting point. And so you can  see at the beginning, it just looks like our  content image. But as we do more and more  steps, we maintain the structure because we're 
still using the content loss as one component  of our loss function. But now we also have 
more and more of the style because of the  early layers, we're evaluating that style 
loss and you can see this doesn't have the  same layout as our spider web, but it has  the same kinds of textures and the same types  of structure there. And so we can check out 
the final result and you can see it's done  ostensibly what our goal is. It's taken one  image and it's done it in the style of  another. And to me, this is quite satisfying. 
JEREMY: And it's actually  done it in a particularly   clever way because look at her arm. Her arm has the spider web nicely laid out on it and  
she's almost picking it out with her fingers. And her face, which is quite important or very  
important in terms of object recognition, the model didn't want to mess with the face  
much at all. So it's kept the spider webs away from that. Like I think it's   the more you look at it, the more impressive it is in how it's managed to find a way to add  
spider webs without messing up the overall kind of semantics of the image. 
JOHNO: Yeah. Yeah. So this is really fun to  play with. If you've been running the notebook 
Possible experiments and miniai
with the demo images, please like, right now,  go and find your own pictures, make sure you're  not stealing someone's licensed work, but  there's lots of creative commons images out 
there and try bash them together. Do it at  a larger size, get some higher resolution 
style loss going. And then there's so much  that you can experiment with. So for example,  you can change the content loss to focus on  maybe an earlier layer as well. You can start 
from a random image instead of the content  image, or you can start from the style image  and optimize towards the content image. You  can change how you scale these two components 
of the loss function. You can change how long  you train for, what your learning rate is.  All of this is up for grabs in terms of what  you can optimize and what you can explore. 
And you get different results with different  scalings and different focus layers. So there's 
a whole lot of fun experimentation to be done  in terms of finding a set of parameters that  gives you a pleasing result for a  given style content pair and for a  
given effect that you want on the output.  TANISHQ: On that note, I wanted to... One of the  really interesting things about this is just how 
well VGG works as a network, even though it's  a very old network. But I think it's also  worth playing around with other networks as  well. I think there's definitely some special 
properties of VGG that allow to do well for  style transfer. And there are a few papers 
on that. And there are also some papers that  explore how we can use maybe other networks  for style transfer that maintains maybe some  of these nice properties of VGG. So I think 
that could be interesting to explore some  of these papers. And of course we have this  very nice framework that allows us to easily  plug and play different networks and try that 
aspect out as well. JEREMY: Yeah. And in particular, I think taking  
a ConvNeXt or a ResNet or something and replacing its head with a VGG head would be an interesting  
thing to try. JOHNO:   Yeah. Tanishq, on the experimentation version,  one of the things that when we were developing 
this I said to Jeremy was like, ah, you're  doing all this work, setting up these callbacks  and things, you know, isn't it nicer to just  have like, here's my image that I'm optimizing, 
set up an optimizer, set up my loss function  and do this optimization loop. And the answer 
is that it is theoretically easier when you  just want to do this once. And that's why 
you see in a tutorial or something, you keep  this as minimal as possible. You just want  to show what style loss is. But as soon as  you say, okay, I'd like to try this again, 
but adding a different layer. So maybe let me  do another cell and then copying and pasting  over a bunch, you know, and then you say, oh,  let me add some progress stuff. So images, 
you know, it gets messy really quickly. As  soon as you want to save images for a video 
and you want to mess with the loss function  and you want to do some sort of annealing  on your learning rate, each of these things  is going to grow this loop into something 
messier and messier. Yeah. And so I thought  it was fun. Like I was very quickly a convert  to being able to experiment with a completely  new version with, yeah, minimal lines of code, 
minimal changes and having everything in its  own piece, like the image logging, or you 
wanted to make a little movie showing the  progress that goes in a separate callback.  You want to tweak the model. You're just taking  one thing that all the other infrastructure 
can stay the same. Yeah. So that was pretty cool. JEREMY: I mean, there's not like one answer,   right? Like it's, yeah. Use  the right layer of abstraction 
for what you're doing at the right time. Like  something I actually think people do too much 
of when they use the fastai library is jumping  straight into data blocks, for example, even 
though the model, you know, they might be  working on a slightly more custom thing where 
data blocks, there isn't a data block already  written for them. And so then step one is 
like, Oh, write a data book. That's not at  all easy. And you actually want to be focusing  on building your model. So I kind of say to  people, Oh, well like, you know, go, go down 
a layer of abstraction. Now I will say I don't  very often start at the very lowest level 
of abstraction. So something like the very  last thing that you showed Johno, just because  in my experience, I'm not good enough to do  that. Right. And so like most of the time 
I yeah, I'll forget zero grad or I'll, I'll  just mess up something, especially if I want 
to like have it run reasonably quickly by  using like, you know, fp16, mixed precision 
or you know, or I'll be like, Oh, now I've  got to like think about how to put a metrics 
in so that I can see it's training properly.  And I always mess that up. And so I don't 
often go to that level, but I do like quite  often start at a reasonably low level. And 
I think with miniai now we all have this  tool where we fully understand all the layers 
and there aren't that many. And yeah, you  could like write your own TrainCB or whatever. 
And at least you've got something that makes  sure, for example, that, Oh, okay. You remember  to use torch.no_grad here. And you remember  to put it in, you know, put it in eval mode 
there that, you know, those things will be  done correctly. And you'll be able to easily 
run a learning rate finder and easily have it  run on CUDA and you know, or whatever device 
you're on. So I think, you know, hopefully  this is a good place for people now to have 
a framework that they can call their own,  you know, and use as much or as little of  it as makes sense. TANISHQ: The other nice thing is of course, like  
there are multiple ways of doing the same thing and it's like whatever way maybe works better   for you. You can implement that. Like for example, Jonathan showed with the ImageOpt  
callback, you know, you could implement that in different ways and whichever one   I guess is easier for you to understand or easier for you to work with, it's, you can implement it  
that way. And yeah, miniai is flexible in multiple ways. So that's the,  
especially one thing I really enjoy about it. JEREMY: Yeah. And this is why extreme of   like weirdness, I think, which  is like, Johno is like using 
miniai for something that we never really  considered making it for, which is like, it's  not even looping through data. It's just looping  through loops. So, you know, this is about 
as weird as it's going to get, I guess. JOHNO: Yeah. Well the next notebook is about   as weird as it's going to get, I think. JEREMY: Oh great. 
Neural Cellular Automata (NCA) notebook
JOHNO: Yeah. Okay. So before we move on to what  we're going to do next is use this kind of style  loss in an even funkier way, to train a  different kind of thing. But before we do 
that, I did want to just call out like using  these pre-trained networks as very smart feature  extractors is pretty powerful. And unlike  the kind of fun, crazy example that we're 
going to look at just now, they also have  very valid uses. So if you're doing like a  super resolution or even something like diffusion,  adding in a perceptual loss or even a style 
loss to your target image, it can improve  things. We've played around with using perceptual  loss for diffusion. Or even during, like say  you want to generate an image that matches 
a face in some kind of image to image thing  with stable diffusion, maybe you have an extra  guidance function that makes sure  that structurally it matches,  
but maybe a textually it doesn't.  Maybe you want to pass in a style image and  have that guide to the diffusion process to 
be a particular style without having to say,  you know, in the style of it's a starry night. 
Yeah. And for all sorts of like image to image  tasks, this perceptual, this idea of like  using the features from a network like VGG,  it does actually have lots of practical uses 
apart from just this artistic and fiddling.  Okay. So speaking of artistic fiddling, we're 
going to look at something a little bit more  niche now called Neural Cellular Automata.  And so try and spend about half an hour on  this before we move on to the next section. 
And so this is off the beaten track. It's  a really fun domain of yeah, like combining 
a lot of different fields, all of which I'm  quite excited about. And so you may be familiar  with like, kind of, classic cellular automata.  And so if we look at Conway's Game of Life, 
oops, I misspelled it, but you've probably  seen this kind of classic… Conway's Game of 
Life. JEREMY: Oh, [unintelligible] there   when I was a kid. JOHNO: Yeah. So the idea here is that you have  
all of these independent cells and each cell can only see its neighbors and you have some sort of  
update rule, right? That says if a cell has three neighbors, it's going to remain   the same state for the next one. If it has only one neighbor, it's going to die in the next  
iteration. And so this is a really cool example of like  
a distributed system, a self-organizing system  where there's no like global communication 
or anything. Each cell can only look at its  immediate neighbors. And typically the rules  are really small and simple. And so we can  use these to model these complex systems. 
It's very much inspired by biology where we  actually do have huge arrangements of cells, 
each of which is only seeing its neighborhood,  like sensing chemicals in the bloodstream  next to it and so on. And yet somehow  they're able to coordinate together. 
JEREMY: I watched a really cool  Kurzgesagt video the other day   about ants and I didn't know this before, maybe everybody else does, but ants like huge  
ant colonies organized by like having little chemical signals that the ants around can  
smell and yeah, it can like organize the entire massive ant colony just using that. I thought it  
was crazy, but it sounds really similar. JOHNO: Yeah. Yeah. And you can do…  
so I’m doing my tangent. You could do very similar 
things where you have, yeah, like the trails,  chemical trails being left, but just like  pixel values in some grid and your ants are  just little tiny agents that have similar 
rules. And so I should probably link this  here, but this is exactly that kind of system,  right? Each little tiny dots, which almost too  small to see is leaving behind these different 
trails. And then that determines the behavior,  the difference between this and what we're  going to do today is that, JEREMY: Sorry, just to clarify, I think you've  
taught me before that like actual slime molds kind of do this right there. They're another example.  JOHNO: Yeah, yeah, exactly. There's some limited  signaling. Each one is like, oh, I'm by food. 
And then after that, that signal is going  to propagate and anything that's moving is  going to follow. And so, yeah, if you play  with this kind of simulation, you often get 
patterns that look exactly like emergent patterns  in nature, like ants moving to food or you 
know, corals coordinating and that sort of  thing. So it's a very biologically biofield. 
The difference with our cellular automata  is that they're going to be, there's nothing  moving. Each, each, like, grind cell has its own  little agent. And so there's no like wandering 
around. It's just each individual cell  looking at its neighbors and then updating.  JEREMY: And just to clarify, when you say agent,  that can be really simple. Like I don't really 
remember, but I vaguely remember that Conway's  Game of Life. It's kind of like a single kind 
of if statement. It's like, if there's, I  don't know, what is it? Two cells around,  you get another one or something. JOHNO: Yeah. Yeah. If there's two or three nearby,  
you stay alive in the next one. If you're overcrowded with four or five or there's   no one near you with zero or one neighbors, then you're going to die. So very, very simple rule.  
But what we're going to do today is replace that hard coded if statement   with a neural network. And in particular, a very small neural network. So I should start with the  
Alexander Mordvintsev’s NCA simulation
paper that inspired me to like even begin looking at this. So this is by Alexander  
Mordvintsev and a team with him at Google Brain. And they built these Neural Cellular Automata.  
So this is a pixel grid. Every pixel is a cellular automata that's looking at its neighbors  
and they can't see the global structure at all. And it starts out with a single black pixel   in the middle. And if you run the simulation, you can see it builds this little lizard  
structure, this little emoji. JEREMY: So that's wild to me   that, that, that a bunch of pixels  that only know about their neighbor 
can actually create such a  large and sophisticated image. 
JOHNO: Yeah. They can self assemble into this.  And what's more, the way that they train them, 
they are robust. They're able to repair damage.  And so there's no, it's not perfect, but there's 
no global signaling. No, no little agent here  knows what the full picture looks like. It  doesn't know where in the picture it is. All  it knows is that its neighbors have certain 
values and so it's going to update itself  to, to match those values. And so you can  see us JEREMY: It  
does seem like something that ought to have  a lot of use in the real world with like, 
I don't know, like having a bunch of drones  working together when they can't contact, 
you know, some kind of central base. So I'm  thinking about like work that Australia, some  Australian folks have been involved in where  they were doing like subterranean, automated 
subterranean rescue operations. And they've  got, you literally can't communicate through  thousands of meters of rock,  stuff like that. JOHNO: Yeah.  
Yeah. So this idea of like self organizing systems, there's a lot of promise for like  
nanotechnology and, and things like that that can do pretty amazing things. This is the   blog post that's linked. Yeah. “The Future of Artificial Intelligence is Self-Organizing  
and Self-Assembling”. And definitely, yeah. It's a pattern that's worked really well in  
nature, right? Like lots of loosely coordinated cells coming together and talking about deep   learning is quite a miracle. And so I think, yeah, that's an interesting pattern to explore.  
Setting up a Neural Network
Okay. So how do we train something like this? How on earth do you set up your  
structure so that you can get something that not only builds out an image or builds out   something like a texture, but then is robust and able to maintain that and keep it going?  
So the sort of base is that we're going to set up a neural network with some learnable  
weights. That's going to apply our little update rule, right? And this can, this is  
just going to be a little dense MLP. We can get our inputs, which is just the neighborhood  
of the cell. And they sometimes have like additional channels that aren't shown that  
the agents can use as communication with their neighbors. So we can, we can set this up in  
code. We'll be able to get our neighbors using maybe convolution or some other method, flatten  
those out and feed them through a little MLP and take our outputs and use that as our   updates. JEREMY: Just to clarify  something that I missed originally 
is: this is not a simplified picture of  it. This is it like that, that 3 by like, 
it's literally 3 by 3. You're only allowed  to see the little things right next  To you or they can be in a different channel.  Exactly. And this paper has this additional 
step of like cells being alive or dead. But  we're going to do one that doesn't even have  that. So it's, it's even simpler than this  diagram. Okay. So to train this, what we could 
do is we could start from our initial states,  apply our network over some number of steps,  look at the final output and compare it to  our targets and calculate on loss. And you 
might think, okay, well, that's pretty cool.  We can maybe do that. And if you run this, 
you do indeed get something that after some  number of steps can learn to grow into something  that looks like your target image. But there's  this problem, which is that you're applying 
some number of steps and then you're applying  your loss after that. But that doesn't guarantee  that it's going to be stable, [noise  of something falling] stable longer  
term. And so we need some additional way to say, okay, I don't just want to grow into this   image. I'd like to then maintain that shape once I have it.  
And the solution that this paper  proposes is to have a pool of training 
examples, right? And we'll see this in code  just now. So the idea here is that sometimes  we'll start from a random state and we'll  apply some number of updates. We'll apply 
our loss function and update our network.  And then most of the time we'll take that  final output and we'll put it back into the  pool to be used again as a starting point 
for another round of training. And so this  means that the network might see the initial 
state and have to produce the lizard. Or it  might see a lizard that's already been produced  and after some number of steps, it still needs  to look like that lizard. And so this is adding 
like an additional constraint that says even  after much more steps, we'd still like you  to look like the final output. And so, yeah,  it's also nice because like I mentioned here, 
initially the model ends up in various incorrect  states that don't look like a lizard, but  also don't look like the starting point. And  it then has to learn to correct those as well. 
So we get this nice additional robustness  from this in addition. And you can see here, 
now they have a thing that is able to grow into  the lizard and then maintain that structure 
kind of indefinitely. And in this paper,  they do this final step where they sometimes  
chop off half of   the image as additional like augmentation. JEREMY: So you could have like a bunch of drawings  
or something that like can only see the ones nearby and they don't have GPS or something and  
no gust of wind could come along and set them off path and they still reconfigure  
themselves. JOHNO: Yeah,   yeah, exactly. Half of them go offline and  run out of battery. That's fine. So a very, 
very cool paper… But you can see this kind  of training is a little bit more complicated  than oh, we just have a network and some target  outputs and we optimize it. So we're not going 
to follow that paper exactly, although it  should be fairly easy to tweak what we have  to match that. We're instead going to go for  a slightly different one by the same authors 
where they train even smaller networks to  match textures. And so you can imagine our 
style loss is going to come in useful here. We'd  like to produce a texture without necessarily  worrying about the overall structure. We just  want the style. And so the same sort of idea, 
the same sort of training, we're going to  start from random and then after some number  of steps, we'd like it to look like our target  style image. And in fact, actually he's a 
spider web, which I hadn't met just until now. JEREMY: And that's the one thing that makes a   texture texture in this case. Is it something you can tile nicely? Is that? 
JOHNO: Yes. Yeah. And so that tiling is going  to come almost for free. So we're going to have  our input, we're going to look at our neighbors,  we're going to feed that through a network 
and produce an output. And every cell is going  to do the same rule, which will work fine 
by default if we set this up without thinking  about tiling at all, except that at the edges, 
when we do like say a convolution to get our  neighbors, we need to think about what happens  for the neighbors of the cells on the edge,  which ones should those be? And by default, 
those will just be padding of zero. And so  those cells on the edge, a) they'll know they're  on the edge. And b) they won't necessarily  have any communication with the other side. 
If we want this to tile, what we're going  to do is we're going to set our padding mode  to circular. In other words, the neighbors  of this top right cell are going to be these 
cells next to it here and these cells down in  the bottom corner. And then, for free, we're 
Getting into code
going to get tiling. Okay. So enough waffle.  Let's get into code. We're going to download 
our style image. Oops. I need to do my inputs.  
This is going to be our target style image. And again, feel free to experiment with your own,   please. We're going to set up a style loss just like we did in lesson 17A.  
The difference being that we're  going to have a batch dimension  to our inputs to this calculate grams function,  which I didn't do in the style transfer example, 
because you're always dealing with a single  image. Everything else is going to be pretty  much the same. So we can set up our style  loss with the target image, and then we can 
feed in a new image or, in this case, a batch  of images, and we're going to get back a loss. 
So we're, we're setting up our, our evaluation.  We would like after some number of steps, 
our output to look like a spider web. Okay.  Let's define our model. And here I'm making 
a very small model with only four channels  and our hidden number of hidden neurons in  the brain is just going to be  eight. You can increase these. 
JEREMY: Something I would be  inclined to do, people might   want to play with in style loss to target is you're giving all the layers the same  
weight. A nice addition would be to have a vector of weights. You could pass in  
an experiment with that. JOHNO: Definitely. All   right. So the, the world in which the  cellular automata are going to live 
is going to be a grid. We're going to have  some number of them. If we call this function,  number of channels and the size, you could  make it non-square if you care about that. 
For our perception in this little diagram here,  we're going to use some hard coded filters 
and you could have these be learned, right?  There'd be additional weights in the neural  network. The reason they're hard coded is  because the people who were working behind 
this paper, they wanted to keep  the parameter counts really low.   Literally like a few hundred parameters total. And also  
they were kind of inspired by the… JEREMY: A few hundred! That's crazy.   Like cause we've been, even our  little Fashion-MNIST models have 
had quite a few million parameters. JOHNO:  Yeah. So this is, yeah, this is a total,  I should have mentioned, that's one of the  coolest things about these systems is they, 
they really can do a lot with very little  parameters. And so these filters that we just  going to hard code are going to be the identity,  right? Just looking at itself and then a couple 
that looking at gradients again, inspired  by biology where even simple cells can sense 
gradients of chemical concentration. So we're  going to have these filters. We're going to  have a way to apply these filters  individually. JEREMY: Just to help  
people understand that that first one, for example, that's a 3 by   3. It's been kind of like visually flattened out, but if you would have kind of lay it out,  
you could see it's an identity matrix. JOHNO:  
Yeah. Anyway, so you can see these filters. This  one is going to sense a horizontal gradient. 
This one is going to sense a vertical gradient.  And the final one is called a Sobel filter. 
Yeah. So we've got some hard coded filters.  We're going to apply them individually to  each channel of the input. And rather than  having a kernel that has separate weights 
for each channel on the input. And so  we can make a grid. We can apply our... 
JEREMY: I haven't seen, I didn't know circular  was a padding mode before. So that just does the  thing you said where it's basically going to  circle around and kind of copy in the thing 
from the other side when you  reach the edge. JOHNO: Yeah.   Yeah. And this is very useful for avoiding issues on the edges with those. You'll see a  
lot of implementations just deal with the fact that they have slightly weird pixels around   the edge and they don't really look into it. This is one way to deal with that. Yeah. Okay.  
So we can make a grid. We can apply our filters to get our model inputs. And this is going to be  
16 inputs, right? Because we have four channels and four filters. 16 inputs,   that's going to be the input to our little brain. And we have this for every location in the grid.  
So now how do we implement that little neural network that we saw? The way it's shown  
in the diagram is it's a dense linear network. And we can set that up. We have a linear layer  
with number of channels by four, which is the number of filters as its number of   inputs. Some hidden number of neurons.  We have a ReLU. We have a second 
linear layer that's outputting one output per  channel as the update. And so if we wanted 
to use this as our brain, what we'd have to do  is we'd have to deal with these extra dimensions. 
So we take our batch by channel by height  and width. We're going to map the batch and  the height and the width all to one dimension  and the channels to the second dimension. 
So now we have a big grid of  16 inputs and lots of examples. 
JEREMY: I don't think we've  seen einops.rearrange before.   So let's put a bit of a bookmark to come back to teach people about that in maybe the next- 
JOHNO: Yeah. Very, very useful function. But it is  a little complicated because we have to rearrange 
our inputs into something that has just 16  features, feed that through the linear layer,  and then rearrange the outputs back to match  the shape of our grid. So you can totally 
do that. And you can see what parameters we  have on our brain. We have an 8 by 16 inputs  and 8 biases for the first layer. And then  we just have a 4 by 8 weight matrix for 
the second linear layer. And I've said bias  equals false because we're having these networks 
propose an update. And if we want them to  be stable, the update is usually going to  be zero or close to it. And so there's no  need for the bias and we want to keep the 
number of parameters as low as possible. That's  kind of the name of the game. And so that's  why we're setting bias equals false. Okay.  So this is one way to implement this. It's 
not particularly fast. We have to do this  reshaping and then we're feeding these examples  through the linear layer. We can cheat by  using convolution. So this might seem like, 
wait, that isn't the linear layer. We're going  to apply this linear network on top of each 
set of inputs. But we can do that by having  a filter size of one, a kernel size of one 
in our convolutional layer. So I have 16 input  channels in my model input here. And I'm going 
to have eight output channels from this first  convolutional layer. And my kernel size is  going to be 1 by 1. And then I have ReLU and  then I have another 1 by 1 convolutional layer. 
And so we can see this gives me the right  shape output. And if I look at the parameters,  my first convolutional layer has 8 by 16 by 1  by 1 parameters in its filters. And so maybe 
spend a little bit of time convincing yourself  that these two are doing the same operation.  JEREMY: Yeah, this definitely is using cheating. I  mean, this is quite elegant. And in languages like 
APL, actually, there's an operation called  stenciling, which is basically the same idea  as this idea of like applying  some computation over a grid. 
JOHNO: And I should mention that convolutions  are very efficient. All of our GPUs and things 
are set up for this kind of operation. And  what makes neural cellular automata quite  exciting is that because we're doing this  convolution, you have an operation for every 
pixel that we're applying, right? This is  looking at the neighborhood and producing  an output. There's no global thing that we  need to handle. And so this is actually exactly 
what GPUs were designed for. They're designed  for running some operation for every pixel  on your screen to render graphics or show you  your video game or make your website scroll 
nice and slick. And so we can take advantage  of that kind of built in bias of the hardware  from doing lots of little operations in parallel  to make these go really, really fast. And 
we'll show I'll show you just now we can run  these realtime in the browser, which is quite  satisfying. Okay. So now that  we have all that infrastructure 
in place, I'm just gonna put it into a class.  My SimpleCA is my cellular automata. We have  our little brain two convolutional layers and  a ReLU. Optionally we can set the weights of 
the second layer to zero again, because you  want to start by being very conservative in  terms of what updates we produce. Not necessarily  necessary, but it does help the training. 
And then in our forward path…  JEREMY: I would be inclined,   I don't know if it matters, but I'd be inclined to put that, I would be inclined to do,  
you know, nn.init constant zero or put that in a no_grad like often initializing things  
without no_grad can cause problems. JOHNO: Okay. And I'll look into that.  
In the forward method, we're  going to apply our filters to  get our model inputs… JEREMY: Oh,  you got .data.zero_. Okay. So that's,  
that's fine. JOHNO: Yeah. I think this is the built in, the built   in method. All right. JEREMY:  Oh, it's the .data, which is 
the thing that makes it. Yeah. You don't need  torch.no_grad because you've got .data. So 
yeah, it's all good. Cool. JOHNO:  Okay. And so the forward is applying  the filters. It's feeding it through the  first convolutional layer, then the ReLU, then 
the second layer. And then it's doing this final  step, which again goes back to the original 
paper somewhere in here, they mentioned that  they are inspired by biology. And one thing 
that you don't have in a biological system  is some sort of global clock where everything  updates at exactly the same second. It's much  more random and organic. Each one is almost 
independent. And so to mirror that, what we  do here is we create a random update mask. 
And if you go in, let's just actually write  this ops. Let's just make, make a, make a 
cell and check that this is what we're doing.  So I'm going to just go b, h, w is 1… just 
to visualize update_rate. There we go. Yeah.  So this is creating this random mask, some 
zeros and some ones according to what our  update rate is. And this is going to determine 
whether we apply this updates to  original input or not. JEREMY:  
It's a lot like dropout. JOHNO: Yeah, exactly. And why this is nice. If you imagine we start   from a perfectly uniform grid and then every cell is running the exact same rule after  
one update, we will still have a perfectly uniform grid. There's no way for there to   be any randomness. And so we can never like break out of that. Whereas once we add this  
random updates or your subset of cells are going to be updated and now there's some   differences, they have different  neighborhoods and things. 
And so then we get this like added randomness  in, and this is very much like in a biological  system, no cell is going to be identical. So  that's a little bit of additional complexity, 
Training
but again, inspired by nature and inspired  by paper. With all of this in place, we can 
do our training. We're going to use the same  dummy dataset idea as before. We are going 
to have a progress callback, which is a lot  of code, but it's all just basically sitting 
around for doing some plotting. So I'm not  going to spend too much time on that. And  then the fun stuff is going to happen in our  training callback. And so now we are actually 
getting deep into the weeds. We're modifying  our prediction function. This is much more  complicated than just feeding a batch of data  through our model. We are setting up a pool 
of grids, right? 256 examples. And these are  all going to start out as just uniforms zeros. 
But every time we call predict, we're going  to pick some random samples from that pool. 
We're occasionally going to reset those samples  to the initial state. And then we can apply 
the model a number of times. And it's worth  thinking here, if we're applying this model 
50 steps, this is like a 50 layer deep model  all of a sudden. And so we start to get some 
of these... JEREMY: Should it be learn.model   rather than self.learn.model? JOHNO: Oh yes, because I already have learn. Nice.  
Yeah. So we've got to just be aware that by applying this a large number of times,   we could get something like gradient exploding and things like that, which we'll deal with a  
little bit later. But we apply the model a large number of steps. Then we put those  
final outputs back in the pool for the next round of training and we store our predictions.   These are the outputs after we've applied a number of steps. And in the loss,  
we're going to use a style loss saying, does this match the style of my target image? And we're  
going to add an overflow loss that penalizes it if the values are out of   bounds. Just to try and... JEREMY: Change self.learn here too. 
Ah yes, thank you. I think I read this before  we changed the... No, because I've got the 
callback there. Okay, my bad. JEREMY: One more   self.learn.preds.clamp and the overflow loss one. JOHNO: Yes, thank you. There we go. And yeah, so  
get_loss is doing a style_loss plus this overflow loss just to keep things from growing   exponentially out of bounds.  Again, something that's quite 
likely to happen when you're applying a large  number of steps. And so we really want to  penalize that. And the final thing is in learn.backward,  
I've added a technique that is probably going to be quite useful in some other places as   well called gradient normalization. And so we're just running through the parameters  
of our model and we are normalizing them. And this means that even if they're really,  
really tiny, really, really large at the end of that multiple number of update steps,   this is kind of a hack to bring this back into control. 
JEREMY: let's put a bookmark to come back  to that as well in more detail. And I guess 
that before_fit, maybe we don't need anymore. JOHNO: Oh right, because this is now default.  
Okay. Oh, so I should have set this running before we started talking. It is going to take a  
little while. But you can see my progress callback here is scatter plotting the loss.  
And the reason you'll see in the callback here, I'm setting the y limits to the minimum  
of the initial set of losses is just because the overflow loss is sometimes so much larger   than the rest of the loss ones, that you get this really bad scaling. So using a log scaling  
and clipping the balance tends to help just visualize what's actually important,   like the overall trend. JEREMY: I guess the  
last benefit of just not run, we can, then we can see it without you running it.  JOHNO: Oh, right. Yeah. Yeah. So you can see  the outputs here. So what I'm visualizing is the 
examples that we've drawn from the pool every  time we're drawing. In this case, I've got  a fixed batch size that should probably be  an argument. But you can take a look at them 
and kind of compare them to the style last  and see initially that I really looked too  similar. After some training, we get some  definite like webby tendencies. And we can 
take this model and then apply it to like a  random grid and log the images every hundred 
steps or whatever. And you can see that starting  from this random position, it quite quickly  builds this pattern that doesn't look perfectly  spider webby. But in its defense, this model 
has 168 parameters. JEREMY: And the tiles.  JOHNO: That to me is like the  magic of these models is that  
even with very few parameters, they're able to do something pretty impressive.   And if you would like, go back up to where we 
define number of channels and the number of  layers. If you, if you give it more channels  to work with 8 or 16 and more hidden neurons,  32 or 64, you still have a tiny model. But 
it's able to capture some, some much nicer.  So I would say, please on the forums, try 
Preview of what’s possible
some larger sizes. I'll also maybe post some  results. And just to give you a little preview 
of what's possible. So I did a project before  using maniai. So the code's a little messy 
and hacky. But what I did was I logged the  cellular automata. Well, maybe I should show 
this. We, this is way outside of the bounds  for this course, but you can write something 
called a fragment shader in WebGL. So this  is designed to run in the browser. It's a 
little program that runs once for every pixel.  And so you can see here, I have the weights  of my neural network. I have sampling the  neighborhood of each cell. We have our filters. 
You have our activation function. This is in  a language called GLSL. We're running through 
the layers of our network and proposing our  updates. And this one here, I just had more, 
I think more hidden neurons, more channels  and optimized with a slightly different loss  function. So it was a style loss plus CLIP to  the prompt, I think, dragon scales or glowing 
dragon scales. And you can see this is running in  realtime or near realtime because I'm recording. 
And it's interactive. You can click to kind  of like zero out the grid and then see it  like rebuild within that. And so in a similar  way with Mordvintsev's report, I'm logging 
these kind of interactive HTML previews. We've  got some videos and just logging the grids 
from the different things. And so you can  see these are still pretty small as far as  these networks go. I think they only have  four channels because I'm working with RGBA 
shaders but quite fun to see what you can do  with these. And if you pick the right style 
images and train for a bit longer and use  a few more channels, you can do some really  fun stuff and you can get really creative  applying them at different scales. Or I did 
some messing around with video, which again, is  just like messing with the inputs to different 
cells to try and get some cool patterns. So yeah,  to me, this is a really exciting, fun niche... 
JEREMY: Amazing JOHNO: Yeah,   I don't know if there's too many practical  applications at this stage, but I'm already  thinking of denoising cellular automata and  stylizing or image restoration cellular automata. 
And you can really have a lot of fun with  this structure. And I also thought it was  just a good demo of how far can we push what  you can do with a training callback to have 
this pool training and gradient normalization  and all these extra things added in. Very, 
very different from, here's a batch of images  and a batch of labels. So I hope you found  that interesting. I'll stop sharing my screen  and then Jeremy, if you have any questions 
or follow ups. No, that's amazing. Thank you so much. I actually   have to go, but that's just one of the coolest things I've seen.

---

# 21

JEREMY: Hello, Johno. Hello, Tanishq.  Are you guys ready for Lesson 21? JOHNO: Ready… 
TANISHQ: Yep, I'm excited. JEREMY: I don't know what I would   have said if you had said no. So good. 
I'm actually particularly excited because I  had a little bit of a big preview of something  that Johno has been working on, which I think  is a super cool demo of what's possible with 
very little code with miniai. So let me turn it over to Johno.  JOHNO: Great. Thanks Jeremy. 
Yeah, so as you'll see when it's back to Jeremy  to talk through some of the experiments and  things we've been doing, we've been using  the Fashion-MNISTdataset at a really small 
scale and really rapidly try out these different  ideas and see some maybe nuances or things  that we'd like to explore further. And so as we were doing that, I started  
to think that maybe it was about time to explore just ramping up the level, like seeing if we can  
go to the next, like slightly larger datasets, slightly harder difficulty,   just to double check that these ideas still hold for longer training runs and yeah,  
different, more difficult data. JEREMY: That's a really good idea   because I feel like pretty confident  that the learnings from Fashion-MNIST 
are going to move across like most of the  time these things seem to, but sometimes 
they don't and it can be very hard to predict. So this seems like a very, a very wise choice.  JOHNO: Yeah. And so, we'll keep wrapping up, but as  
a next step, one above Fashion-MNIST, I thought I'd look at this data called CIFAR-10. 
And so CIFAR-10 dataset is a very popular dataset  originally for things like image classification, 
but also now for all of this genera–  any paper on generative modeling.  And it's kind of like the smallest  dataset that you'll see in these papers. 
And so, yeah, if you look at  the classification results,   for example, pretty much every classification paper since they started tracking has reported  
results on CIFAR-10 as well as their larger datasets and likewise with image generation,  
very, very popular. All of the recent   diffusion papers will usually report  CIFAR-10 at the end maybe ImageNet 
and then whatever large, massive  dataset they're training on.  JEREMY: So we were, we were somewhat  notable in 2018 for managing to train. 
So for CIFAR-10, 94% classification  is kind of the benchmark.  So there was a competition a few years ago  where we managed to get to that point at a 
cost of like 26 cents worth of AWS time, I  think, which won a big global competition. 
So I actually hate CIFAR-10, but we had  some real fun with it a few years ago. 
JOHNO: Yeah. And it's good.  It's a nice dataset for quickly testing things  out, but we'll talk about why we also like, 
us as a group, don't like it at all. And we'll pretty soon move on to something better. 
The notebook
So one of the things you'll notice in this  notebook, I'm basically using all of the same  code that Jeremy's going to  be looking at and explaining. 
So I won't go into too much, but  the datasets also on HuggingFace.  So we can load it just like  we did the Fashion-MNIST. 
The images are three channels  rather than single channel.  So the shape of the data is slightly  different to what we've been working with. 
That's weird. Yeah.  So we have, instead of a single channel image,  we have a three channel red, green and blue 
image. And this is what a batch of data looks like.  JEREMY: And you've got 32 images in your batch. So that's Batch by Channel by Height by  
Width, right? JOHNO: Yeah.  Yeah. Batch by Channel by Height and Width.  JEREMY: That was a little  confused by the 32 by 32. 
JOHNO: Oh yeah. JEREMY: I got it now. JOHNO: Batch size can be arbitrary.  So if you plot these, one of the things, if  you look at this, okay, I can see these are 
different classes. Like I know this is an airplane, a frog,   an airplane, but it's actually a puzzle with an airplane on the cover, a bird, a horse, a car. 
That one, you squint, you can tell  it's a bear, but only if you really   know what you're looking for.  And so when we started to talk about generating  these images, this is actually quite frustrating. 
Like this, if I generated this, I'd say this  might be the model doing a really bad job. 
And, but it's actually that  this is a boat, this is a dog.  It's just that this is what the data looks like. JEREMY: And so I've actually got something  
that can help you out. I'll show later today,   which is something like this. It's really actually hard to see  
whether it's good because the images are bad. It can be helpful to have a metric to  
generate that can see how good samples are. So I'll be showing a metric for that later today.  JOHNO: Yeah. And that'll be great. 
And I hope to have like automated, but anyway, I  just wanted to flag like for visually inspecting  these it's not great. And so we don't really  
like CIFAR-10 because it's hard to tell. And but still a good, a good one to test with. 
So the noisify and everything I'm following  what Jeremy is going to be showing exactly  the code works without any changes because  we're adding random noise in the same shape 
as our data. So even though   our data now has three channels, the  noisify function still works fine. 
If we try and visualize the noisified images  because we're adding noise in the Red, Green,  Blue channels. And some of that's, you know,  
kind of extreme values. Yeah.  It looks slightly different. Looks all crazy RGB.  But you can see, for example, this frog doesn't  have as much noise and it's vaguely visible. 
But it is, it's a, it's a many impossible  tasks to look at this and tell what image  is hiding under all of that noise. JEREMY: So I think this is really neat that  
you could use the same noisify. Yeah. 
And it's still, it still works. It's not just that shape thing, but I guess  
just thanks to kind of PyTorch's broadcasting kind of stuff, this often happens. 
You can kind of change the dimensions  of things and it just keeps working.  JOHNO: Exactly. And we've been paying  
attention to those broadcasting rules  and the right dimensions and so on.  Cool. So I'm going to use the same sort of approach  
to loading the UNet, except that now obviously I need to specify three input channels and three  
output channels because we're working with three channel images.  But I did want to explore for this demo, like,  okay, how could I maybe justify wanting to 
do this kind of experiment tracking  thing that I'll talk about?  And so I'm bumping up the size  of the model substantially. 
I've gone from, this is the default settings  that we were using for Fashion-MNIST, but  the diffuser's default UNet has what  many, 20 times as many parameters,  
274 million versus 15 million.  So we're going to try a larger model. We're going to try some longer training. 
And so I could just do the same training that  we've always done just in the notebook, set 
up a Learner with ProgressCB to kind  of plot the loss, track some metrics. 
But yeah, I don't know about you, but once  it's beyond a few minutes training, I quickly 
get a patient and I have to wait for  it to finish before we can sample.  So I'm doing the DDPM sample, but I have to,  I actually interrupted the training to say, 
I just want to get a look at what it looks  like initially and to plot some samples.  And again, the sampling function works without  any modification, but I'm passing in my size 
to be a three channel image. Yeah.  And so this is like, we could do it like this,  but at some point I would like to a) keep 
Experiment tracking and W&B callback
track of what experiments I've tried and b)  be able to see things as it's going over time, 
including like, I would love to see what the  samples look like if you generate it after  the first epoch, after the second epoch. And so that's where my like little callback  
that I've been playing with comes in. JEREMY: So just before you do that,   I'll just mention, like, I mean,  there are simple ways you could 
do that, right? Like, you know, one popular way a lot of   people do is that they'll save some sample images as files every epoch or two, or we could like  
the same way that we have a updating plot as we train with fastprogress, we could have  
an updating set of sample images. So there's a few ways we could,  
we could solve that. That wouldn't handle the tracking that  
you mentioned of like looking over time at how different changes have improved things or   made them worse, whatever that would, I guess would require you kind of like saving multiple  
versions of a notebook or keeping some kind of research journal or something.  That'd be a bit fiddly. JOHNO: It is. 
And all of that's doable, but I also find  like I'm a little bit lazy sometimes.  Maybe I don't write down what I'm trying or  yeah, I've saved untitled number 37 notebook. 
So yeah, the idea that I wanted to show here  is just that there are lots of other solutions  for this kind of experiment tracking and logging. And one that I really like is  
called Weights and Biases (W&B). So I'll explain what's going on in the code here,   but I'm running a training with this additional Weights and Biases callback. 
And what it's doing is it's allowing  me to log whatever I'd like.  And so I can log samples at different. JEREMY: Okay. 
So just switching to a  website here called wandb.ai.  So that's where your callback  is sending information to. 
JOHNO: Yeah. So Weights and Biases   accounts are free for personal and academic use. And it's very, very like, I don't think I know  
anyone who writes Weights and Biases, but it's a very nice service.  You sign in and you log in on your computer  or you get a little authentication token. 
And then you're able to log these experiments  and you can log into different projects. 
What it gives you is for each experiment,  anything that you call Weights and Biases  dot log at any step in the training, that's  getting logged and sent to their server and 
stored somewhere where you can  later like access it and display it.  They have these plots that you can, you know,  visualize easily and you can also share them 
very easily and like these reports that  integrate this data sort of interactively. 
And why that's nice is that later, like you  can go and look at, so this is another project  that I'm logging to. You can log multiple runs with different settings. 
And for each of those, you have all of these  things that you've tracked, like your training 
loss and validation, but you can also track  your learning rate to keep doing a learning  rate schedule. And you can save your model as an  
artifact and it'll get saved on their server. So you can see exactly what run produced,   what model. It logs the code. 
If you set that to, you can save,  you know, save code equals true.  And then it creates a copy of your whole Python  environment, what libraries were installed, 
what code you ran. So for being able to come back later and say,   oh, these images here, these look really good. I can go back and see, oh,  
that was this experiment here. I can check what settings I used. 
In the initialization, you can log whatever  configuration details you'd like and any comments. 
And yeah, there's other frameworks for this. JEREMY: Yeah, in some ways it's kind of, initially  
when I first saw Weights and Biases, it felt a bit weird to me actually, like sending your  
information off to an external website because, I mean, before Weights and Biases existed,   the most popular way to do this was something called TensorBoard, which Google provides,  
which is actually a lot like this, but it's a little server that runs on your computer. 
And so like when you log things, it just puts  it into this little database on your computer, 
which is totally fine. But I guess actually there are some benefits to  
having somebody else run this service, you know, instead of running your own little   TensorBoard or whatever server, you know, one is that you can have multiple  
people working on a project collaborating. So I've done that before where we will each be  
sending like different sets of hyper parameters and then they'll end up in the same place.  Or if you want to be really antisocial, you  know, you can interrupt your romantic dinner 
and look at your phone to see  how your training's going.  So like, yeah, I'm not going to say it's like  always the best approach to doing things, 
but I think there's definitely  benefits to using this kind of service.  And it looks like you're showing us that you  can also create reports for sharing this, 
which is also pretty nifty. JOHNO: Yeah.  Yeah. I mean, like for working with other  
people or like you want to show somebody the final results and being able to, yeah, like pull  
together the results from some different runs or just say, oh, look, by the way, here's a  
set of examples from my two most recent and things tracked at different steps. 
What do you think of this? And yeah, being able to have this like   place where everyone can go and they can inspect the different loss curves for any run. 
They can say, oh, you know, what were  the, what was the batch size for this?  Let me go look at the info there. Okay. 
I didn't log it, but I logged how  many epochs and the learning rate.  So yeah, I find it quite nice, especially  in a team or if you're doing lots and lots 
of experiments to be able to like have this  permanent record that somebody else deals  with and may host the storage and the tracking. Yeah, it's quite nice. 
JEREMY: Wait, and this is all  the code you had to write?.  That's amazing. JOHNO: Yeah.  So this is using the callback system. The way Weights and Biases works is that  
you start a… you start an experiment with this wandb.init, and you can specify any like  
configurational settings that you use there.  And then anything you need to log is  wandb.log and you pass in whatever the 
name of your body is. So again, logging the loss and then the value   and once you've done, wandb.finish and that syncs everything up and sends it  
to the server. JEREMY: Oh,   this is wild the way you've inherited  from MetricsCB and you replaced that _log 
that we previously were used to allow  fastprogress to do the logging and you've  replaced it to allow Weights  and Biases to the logging. 
So yeah, it's really sweet. JOHNO: Yeah.  Yeah. So this is using the callback system. 
I wanted to do the things that MetricCV normally  does, which is tracking different metrics  that you pass in. So this will still do that. 
And I just offload to the super, like the  original MetricsCV method for things like  the after_batch. But in addition to that,  
I'd also like to log the Weights and Biases. And so before I fit, I initialize the experiments. 
Every batch I'm going to load  the loss. After every epoch  the default metrics callback is going  to accumulate the metrics and so on. 
And then it's going to call this _log function. So I chose to modify that to say, I'm going to  
log my training loss. It's training.  I'm going to log my validation loss if I'm  doing validation and I'd like it to log some 
samples and Weights and Biases is quite  flexible in terms of what you can log.  You can create images or  videos or audio or whatever. 
But it also takes a matplotlib figure. And so I'm generating samples and plotting  
them with show_image and spitting back that matplotlib figure, which I can then log.  And that becomes these pretty  pictures that you can see over time. 
Like every time that log function runs, which  is after every epoch, you can go in and see 
what the images look like. JEREMY: So where do you think you can   make your code even simpler in the future? If we had show_images, maybe it could have  
like a optional return fig parameter that returns the figure and then we could replace those  
four lines of code with one, I suspect. JOHNO: Yeah.  Yeah. And I mean this,   I just sort of threw this together. It's quite early still. 
You could also, what I've done in the past  is usually just create a PIL image where you 
can, you know, make a grid or overlay  text or whatever else you'd like.  And then just log that as wandb.image. Otherwise like apart from that, I'm just  
passing in this callback as an extra callback to, um, my set of callbacks for the learner,  
instead of a metric callback. And so when I call .fit,   I still get my little progress bar. I still get this printed out version because  
Fitting
my log function still also prints those metrics just for debugging. 
But instead of having to like watch the progress  in the notebook, I can set this running disconnect  from the server, go have dinner and then  I can check on my phone or whatever. 
What did the circles look like? And okay, cool.  It's starting to look like less than random  nonsense, but still not necessarily recognizable. 
Maybe we need to train for longer. That can be the next experiment.  What I should probably do next  is think of some extra metrics,  
but Jeremy's going to talk about that.  So for now, that's pretty much all I had to  show is just to say, yeah, it's worth as you 
move to these longer, you know, 10 minutes,  one hour, 10 hours, these experiments, it's  worth setting up a bit of infrastructure for  yourself so that you know what were the settings 
I used. And maybe you're saving the   model so you have the artifact as a result. And yeah, I like this Weight and Biases  
approach, but there's lots of others. The main thing is that you're doing something   to track these experiments beyond just, you know, creating plenty of different versions  
of your notebook. JEREMY: I love it.  TANISHQ: One thing I was going to note that I  don't know if many people know, but like Weights 
Comments on experiment tracking
and Biases can also save the exact code  that you use to run that for that run. 
So like if you make any changes to your code  and then you, you know, then you don't know  which version of your code you use  for this particular experiment. 
So then you can figure out  exactly what code you use.  So it's all completely reproducible. And so I love, you know, Weights and  
Biases, all these different features it has.  And I use Weights and Biases all the time  for my own research, like almost daily. 
Like I had to put a run just last  night and check on it today morning.  So it's like, I use it all the time for my own  research and yeah, like I use it especially 
to just know like, oh, this  run had this particular config.  And then like, yeah, the models go  straight into Weights and Biases. 
And then if I want to run a model on the test  set, I literally actually take it off of Weights  and Biases, like download it from Weights  and Biases and run it on the test set. 
So I use it all the time. And also just having the ability   to have everything reproducible  and know exactly what you were 
doing is very convenient instead of having  to like manually track it in some sort of  like, I guess a big Excel sheet or some  sort of journal or something like that. 
Sometimes this is, you know, this  is a lot more convenient, I feel so.  Yeah. JEREMY: Last we get into too much chilling from  
Weights and Biases. I'm going to put a slightly alternative point of view, which is, I don't use  
it or any experiment tracking framework myself.  Which is not to say maybe I could get some  benefits by doing so, but I fairly intentionally 
don't because I don't want to make it easy  for myself to try a thousand different hyper 
parameters or do kind of like   all-directed, you know, sampling of things. I like to be very like directed, you know. 
And so that's kind of the workflow I'm  looking for is one that allows that to happen, 
right?, constantly go back  and refactoring and thinking,   what did I learn and how do I change things from here and never kind of doing like 17 learning  
rates and 6 architectures and whatever. Now obviously that's not something  
that Johno is doing at the moment. And it’ll be so easy for him to get on,  
if you wanted to. JOHNO: I can now make   a script that just does a hundred runs  with different models and different tasks 
and then I can look at my Weight and  Biases and say filter by the best loss.  JEREMY: Yeah. JOHNO: Which is very tempting.  JEREMY: I would say to people like, yeah,  definitely be aware that these tools exist. 
And I definitely agree that as we do this,  which is early 2023, Weights and Biases is 
by far the best one I've seen. It has by far the best integration with fastai.  And as of today, if Johno's pushed yet, it  has by far the best integration with miniai. 
I think also fastai is the best library  for using with Weights and Biases.  It works in both ways. So yeah, know it's there. 
Consider using it, but also consider not going  crazy on experiments because, you know, I 
think experiments have their place clearly,  but also carefully thought out hypotheses, 
testing them, changing your code is  overall the approach that I think is best. 
Well, thank you, Johno. I think that's awesome. 
I got some fun stuff to share as  well, or at least I think it's fun. 
FID and KID, metrics for generated images
And what I wanted to share is like, well…  first of all, I should say we had said, 
we all had said that we were  going to look at UNets this week.  We are not going to look at UNets this week,  but we have good reason, which is that we 
had said, we're going to go from  foundations to Stable Diffusion.  Well, that was also a lie because we're  actually going beyond Stable Diffusion. 
And so we're actually going to start  showing today some new research directions.  I'm going to describe the process  that I'm using at the moment to  
investigate some new research directions.  And we're also going to be looking at some  other people's research directions that have 
gone beyond Stable Diffusion  over the past few months. 
So we will get to UNets, but we haven't quite  finished, you know, as it turns out, the training 
and sampling yet. Now   one challenge that I was having as I started  experimenting with new things was started 
getting to the point where actually the generated  images looked pretty good and it felt like 
you know, almost like being a parent, you  know, each time a new set of images would 
come out, I would want to convince myself  that these were the most beautiful.  And so I, yeah, I, I, I, I, and like when  they're crap, it's obvious they're crap, you 
know, but when they're starting to look pretty  good, it's very easy to convince yourself  you're improving. So I wanted to, you know, have a  
metric which could tell me how good they were. Now unfortunately there is no such metric. 
There's no metric that actually says, do these  images, would these images look to a human 
being like pictures of clothes? Because only talking to a person can do that.  But there are some metrics which  give you an approximation of that. 
And as it turns out, these metrics are not  actually, they're not actually a replacement 
for human beings looking at things. But they're a useful addition. 
So I mean, I certainly found them useful. So I'm going to show you the two most common,   well there's really the one most common metric, which is called FID. 
And I'm going to show another  one called KID or KID. 
FID notebook (18_fid.ipynb)
So let me describe and show how they work. And I'm going to demonstrate them using  
the model we trained in the last lesson, which was in DDPM2. 
And you might remember we  trained one with mixed precision  
and we saved it as fashion_ddpm_mp for mixed precision. 
Okay, so this is all the usual imports and stuff. This is all the usual stuff. 
But there's a slight difference this time,  which is that we're going to try to get the  FID for a model we've already trained. So basically to get the model we've already  
trained to get its FID, we can just torch.load it and then dot cuda() to pop it on the GPU. 
So I'm going to call that the smodel, which  is the model for samples, the samples model.  And this is just a copied and  pasted DDPM from the last time. 
So that's for sampling. So we're going to do sampling from that model. 
And so once we've sampled from the model,  we're then going to try and calculate this 
score called the FID. Now what the FID is going to do is it's  
not going to say how good are these images. It's going to say   how similar are they to real images. And so the way we're going to do that is  
we're going to actually look specifically at for the images that we generated in these samples,  
we're going to look at some statistics of some of the activations. 
So what we're going to do, we've generated  these samples and we're going to create a 
new Dataloaders, which  contains no training batches. 
And it contains one validation  batch, which contains the samples.  It doesn't actually matter  what the dependent variable is. 
So I just put in the same dependent  variable that we already had.  And then what we're going to do is we're going  to use that to extract some features from 
a model. Now, what do we mean by that?  So if you remember back to notebook 14, we  created this thing called summary and summary 
shows us at different blocks of our model,  there are various different output shapes. 
In this case, it's a batch size of 1024. And so after the first block,   we had 16 channels, 28 by 28. And then we had 32 channels, 14 by 14  
and so forth until at the, just before the final linear layer, we had, we had the 1024 batches and  
we had 512 channels with no height and width.  Now the idea of FID and KID is that  the distribution of these 512 channels  
for a real image has a particular kind of like signature, right? 
It looks a particular way. And so what we're going to   do is we're going to take our samples. We're going to run it through a model that's  
learned to predict, you know, fashion classes, and we're going to grab this layer, right? 
And then we're going to average  it across a batch, right?  To get 512 numbers. And that's going to  
represent the mean of each of those channels. So those channels might represent, for example,  
you know, does it have a pointed collar? Does it have, you know, smooth fabric? 
Does it have sharp heels and so forth? Right.  And you could recognize that something is  probably not a normal fashion image if it 
says, oh yes, it's got sharp  heels and flowing fabric. 
It's like, oh, that doesn't  sound like anything we recognize.  So there are certain kind of like sets of  means of these activations that don't make 
sense. So that's...  JOHNO: This is a metric for, it's not a metric  for an individual image necessarily, but it's 
across a whole lot of images. So if I generate a bunch of fashion images   and I want to say, does this look like a bunch of fashion images? 
If I look at the mean, like maybe X percent  have this feature and X percent have that 
feature. So if I'm looking at those means, as   like comparing the distribution, within all these images I generated, do roughly the same  
amount have sharp collars as those in the… JEREMY: Yeah, that's a very good point too. 
JOHNO: Well, the features generated are similar. JEREMY: Yeah.  And it's actually going to get  even more sophisticated than that,   but let's just start at that level, which is this features dot mean. 
So the basic idea here is that we're going  to take our samples and we're going to pass 
them through a pre-trained model that has  learned to predict what type of fashion something 
is. And of course we train some of those in this  
notebook. And specifically we trained a nice 20 epoch   one in the data augmentation section, which had a 94.3% accuracy. 
And so if we pass our samples through this  model, we would expect to get some useful 
features. One thing that I found made this a bit complicated   though, is that this model was trained using data that had gone through this transformation  
of subtracting the mean and dividing by the standard deviation.  And that's not what we're creating in our samples. And so generally speaking, samples in most of  
these kinds of diffusion models tend to be between negative one and one.  So I actually added a new section to the very  bottom of this notebook, which simply replaces 
the transform with something that goes from  negative one to one, and just creates those 
Dataloaders and then trains  something that can classify fashion. 
And I saved this as not data_aug, but data_aug2. So this is just exactly the same as before,  
but it's a fashion classifier where the inputs are expected to between minus one and one. 
Having said that, it turns out that our  samples are not between minus one and one. 
But actually, if you go  back and you look at DDPM2,  
we just use TF.to_tensor, and that actually makes images that are   between zero and one. So actually that's a bug. 
Okay, so our images have a bug, which  is they go between zero and one. 
So we'll look at fixing that in a moment. But for now, we're just trying to get the   FID of our existing model. So let's do that. 
Get the FID from an existing model
So what we need to do is we need to take the  output of our model, and we need to multiply 
by two, so that'll be between  zero and two, and subtract one.  So that'll change our samples to be between  minus one and one, and we can now pass them 
through our pre-trained fashion classifier.  Okay, so now how do we get that the output of  that pooling layer, because that's actually 
what we want to remind you. We want the output of  
this layer.  So just to kind of flex our PyTorch muscles,  I'm going to show a couple of ways to do it. 
So we're going to load the model I  just trained, the data_aug2 model.  And what we could do is, of  course, we could use a hook. 
And we have a hook's callback. So we could just create a function,   which just appends the output. So very straightforward. 
Okay, because that's what we want.  We want the output. And specifically, it's,  
so we've got these are all sequentials. So we can just go through and go, oh, one, two,  
three, four, five, the layer that we want.  Okay, and so that's the  module that we want to hook. 
So once we've hooked that, we  can pass that as a callback.  And we can then, it's a bit weird calling  fit, I suppose, because we're saying train 
equals false, but we're just basically capturing. This is just to put make one_batch go through  
and grab the outputs. So this means now in our hook, there's never  
going to be thing called outp, because we put it there.  And we can grab, for example,  a few of those to have a look. 
And yep, here, we've got a  64 by 512 set of features. 
Okay, so that's one way we can do it. Another way we could do it is that actually  
sequential models are what's called in Python collections, they have certain,  
a certain API that they're expected to support. And out of something a collection can do like a  
list is you can call del to delete something. So we can delete this layer and  
this layer and be left with just these layers. And once we do that, that means we can just call  
capture_preds, because now they don't have the last two layers.  So we can just delete layers eight  and seven, call capture_preds. 
And one nice thing about this is  it's going to give us the entire  
10,000 images in the test set.  So that's what I ended up deciding to do. There's lots of other ways I played around  
with which worked, but I decided to these two as being two good, pretty good techniques. 
Okay, so now we've got what do 1,000 real images  look like at the end of the pooling layer. 
So now we need to do the same for our sample. So we'll load up our fashion_ddpm_mp,  
we’ll call sample. Let's just grab 256 images for now, make them  
go between minus one and one, make sure they look okay.  And as I described before, created Dataloaders  where the validation set just has one batch, 
which contains our samples and call capture_preds. Okay, so that's going to give us our features. 
And the reason why is because we're passing the  sample to model and model is the classifier. 
Okay, which we've deleted  the last two layers from.  So that's going to give us our 256 by 512. So now we can get the means. 
Now that's not really enough to tell us  whether something is looks like real images. 
So maybe I should draw here. So we started out with our  
batch of 256 and our channels of 512. And we squished them by taking their mean. 
It's now just 256 vector. So this is the wrong way around. 
We squished it this way, 512, because  this is the main for each channel. 
And we did exactly the same  thing for the much bigger,  
full set of real images. So this is our samples.  And this is our real. 
But when we squish it, that's  10,000 by 512, we get again 512. 
So we could now compare these two. But you know, you could absolutely have some  
samples that don't look anything like images, but have similar averages for each channel. 
Covariance matrix
So we do a second thing, which  is we create a covariance matrix. 
Now if you've forgotten what this is, you  should go back to our previous lesson where  we looked at it. But just to remind you, a covariance matrix  
says, in this case, we do it across the channels, it's going to be 512 by 512. 
So it's going to take each of these columns. And it says, in each cell, so here's cell 1, 1. 
Basically it says, what's the difference between…  it basically saying what's the difference 
between each row, each element here, and the  mean of the whole column, multiplied by the 
exactly the same thing for a different column. Now on the diagonal, it's the same column twice.  So that means that these in the  diagonal is just the variance. 
But more interestingly, the ones in the off  diagonal, like here, is actually saying, what's 
the relationship between  column one and column two.  So if column one and column two are  uncorrelated, then this would be zero. 
If they were identical, then it would  be the same as the variance in here. 
So it's how correlated are they. And why is this interesting?  Well, if we do the same, exactly the same  thing for the reals, that's going to give 
us another 512 by 512. And it's going to say things like, so let's  
say this first column was kind of like, you know, does it have pointy heels? 
And sorry, heels, can't spell. And the second one might be,  
does it have flowing fabric? Right.  And this is where we say, okay, if, you know,  generally speaking, you would expect these 
to be negatively correlated. Right?  So over here in the reals, this is  probably going to have a negative, right? 
Whereas if over here it was like zero or even  worse if it's positive, it'd be like, oh,  those are probably not real, right? Because it's very unlikely you're going  
to have images that have both where pointy heels are positively associated with a flowing fabric. 
So we're basically looking for two data sets  where their covariance matrices are kind of 
the same and their means  are also kind of the same. 
All right. So there are ways of  
comparing these, you know,  basically comparing two sets of data 
to say, are they, you know,  from the same distribution?  And you can broadly think of it as being like,  oh, do they have pretty similar covariance 
matrices? Do they have pretty similar mean vectors?  And so this is basically what the  Fréchet Inception Distance does. 
Does that make sense so far, guys? JOHNO: Yes. 
It's only striking me now how strong the  similarity as to when we're talking about 
like the style loss and those kinds of things.  How do we measure the types of features that  occur together without worrying about like 
which items in the data set? JEREMY: The Gram-Schmidt matrices or whatever.  JOHNO: Exactly. JEREMY: Yeah. 
JOHNO: Yeah. JEREMY: Now  
the particular way of comparing, so, okay, so  I've got the means and I've got the covariances 
for my samples. And I've actually  
just created this little _calc_stats, right? So I always, I'm showing you how I build things,   not just things that are built, right? So I always create things step by  
step and check their shapes, right? And then I paste them into our, merge the cells,   copy the cells and merge them into functions. 
So here's something that gets the  means and the covariance matrix. 
So then I basically do, I call that both for  my sample features and for my features of 
the actual data set or the  test set and the data set. 
Now what I now do with that, if they have  those features, I can calculate this thing  called the Fréchet Inception  Distance, which is here. 
And basically what happens is we multiply  together the two covariance matrices and that's 
now going to make them like bigger, right? So we now need to basically scale that down again. 
Matrix square root
Now, if we were working with, you know, non  matrices, you know, if you kind of like multiply 
two things together, then to kind of bring  it back down to the original scale, you know, 
you could kind of like take  the square root, right?  So particularly if it was by itself, you took  the square root, you get back to the original. 
And so we need to do exactly the same  thing to renormalize these matrices. 
The problem is that we've got matrices and  we need to take the matrix square root. 
Now the matrix square root, you might not  have come across this before, but it exists.  And it's the thing where the matrix square  root of the matrix A times itself is A. Now 
I'm going to slightly cheat because   we've used the float square root before and we did not re-implement it from scratch  
because it's in the Python standard library. And also it wouldn't be particularly interesting,   but basically the way you can calculate the float square root from scratch is by using,  
there's lots of ways, but you know, the classic way that you might've done it in high school is   to use Newton's method, which is where you basically can solve, if you're  
trying to calculate, you know, A equals the root X, then you're basically saying A squared equals  
X, which means you're saying A squared minus X equals zero. 
And that's an equation that you can solve  and you can solve it by basically taking the 
derivative and taking a step along  the derivative a bunch of times. 
You can basically do the same thing  to calculate the matrix square root.  And so here it is, right? It's the Newton method,  
but because it's the matrices,  it's slightly more complicated.  So it's the Newton-Schurz method and I'm not  going to go through it, but it's basically 
the same deal you go through up to a hundred  iterations and you basically do something 
like traveling along that kind of derivative. And then you say, okay, well, the result times  
itself ought to equal the original matrix. So let's subtract the matrix times itself from  
the original matrix and see whether the absolute value is small and if it is,   we've calculated it. Okay. 
So that's basically how we  do a matrix square root.  So we do that. And so now that we have, strictly speaking,  
implemented from scratch, we're allowed to use the one that already exists.  PyTorch doesn't have one, sadly, so we have  to use the one from SciPy, scipy.linalg 
So this is basically going to give us a measure  of similarity between the two covariance matrices 
and then we, here's the measure  of similarity between the two  
main matrices, which is just the sum of squared errors.  And then basically for reasons that aren't  interesting, but it's just normalizing, we 
subtract what's called the trace, which  is the sum of the diagonal elements.  And we subtract two times  the trace of the, this thing. 
And that's called the Fréchet Inception Distance. Also a bit hand-wavy on the math because I  
don't think it's particularly relevant to anything, but it gives you a number which  
represents how similar is, you know, this for the samples to this for some real data. 
Now it's weird it's called Fréchet Inception  Distance when we've done nothing to do with 
Why it is called Fréchet Inception Distance (FID)
inception. The reason why is that people do not normally use  
the fastai Part 2 custom Fashion- MNIST data_aug2 dot pickle. 
They normally use a more famous model. They normally use the inception model,  
which was an ImageNet winning model from Google Brain from a few years ago. 
There's no reason whatsoever that  inception is a good model to use for this. 
It just happens to be the one  which the original paper used.  And as a result, everybody now uses that not  because they are sheep, but because you want 
to be able to compare your results  with other people's results.  Perhaps we actually don't, we actually want  to compare our results from our other results. 
And we're going to get a much more accurate  metric if we use a model that's good specifically 
at recognizing fashion. So that's why we're using this.  So very, very few people bother to use this. Most people just pip install Python FID or  
whatever it's called and use inception, but it's actually better to use.  Unless you're comparing to papers, it's better  to use a model that you've trained on your 
data and you know is good at that. So I guess this is not a FID. 
It's a, well maybe FID now  stands for Fashion-MNIST.  I don't know what it stands for. I should have done something. 
Some FID caveats
TANISHQ: I wanted to bring up two other  caveats of FID, especially in papers. 
The other thing is that FID is dependent  on the number of samples that you use. 
So as the number of samples they use for measuring  FID, it's more accurate if you use more samples 
and it's less accurate if you use less samples. JEREMY: What I think is less accurate is actually   biased. So if you use less samples,  
it's too high specifically. TANISHQ: Yeah.  So in papers, you'll see them  report how many samples they used. 
And so even then comparing to other papers  and comparing between different models and  different things, you want to make sure that  you're comparing with the same amount of samples. 
Otherwise it might just be high because they  just use less number of samples or something  like this. So you want to make sure that's comparable. 
And then the other thing that is, because I  guess it's a kind of a side effect of using 
the Inception network in these papers is the  fact that all of these are at a size 299 by 
299, which is like the size that  the Inception model was trained.  So actually when you're applying this Inception  network for measuring this distance, you're 
going to be resizing your images to 299 by  299, which in different cases that may not 
make much sense. So like in our case,   we're working with 32 by 32 or 28 by 28 images. These are very small images. 
And if you resize it to 299 or in other cases,  this is now kind of an issue with some of 
these latest models, you have these  large 512 by 512 or 1024 by 1024 images. 
And then you're kind of shrinking these images  to 299 by 299 and you're losing a lot of that 
detail and quality in those images.  So actually it's kind of become a problem  with some of these latest papers when you 
look at the FID scores and  how they're comparing them.  And then visually when you see them, you can  kind of notice, oh yeah, these are much better 
images, but the FID score doesn't capture  that as well because you're actually using  these much smaller images. So there are a bunch of different caveats. 
And so FID, it's very good for like, yeah,  it's nice and simple and automated for this 
sort of comparison, but you have to be aware  of all these different caveats of this metric  as well. JEREMY: So excellent segue because we're going  
KID: Kernel Inception Distance
to look at exactly those two things right now. And in fact, there is a metric that compares the  
two distributions in a way that is not biased.  So it's not necessarily higher or lower if  you use more or less samples and it's called 
the KID or KID, which is the  Kernel Inception Distance.  It's actually significantly simpler to  calculate than the Fréchet Inception Distance. 
And basically what you do is you create a  bunch of groups, a bunch of partitions, and 
you go through each of those partitions and  you grab a few of your Xs at a time and a 
few of your Ys at a time. And then you calculate something called the MMD,  
which is here, which is basically the… again, the details don't really matter. 
We basically do a matrix product  and we actually take the cube of it. 
This is this K is for kernel, and we basically  do that for the first sample by its compared 
to itself, the second compared to itself  and the first compared to the second. 
And we then normalize them in various ways  and add the two with themselves together and 
subtract the, with the other one. And this one actually does not use the stats. 
It doesn't use the means and covariance metrics. It uses the features directly. 
And the actual final result is basically the  mean of this calculated across different little 
batches. Yeah, again, the math doesn't really   matter as to, you know, exactly why all these are exactly what they are, but it's going to give you  
again, a measure of the similarity of these two distributions. 
At first I was confused as to why more people  weren't using this because people don't tend 
to use this and it doesn't  have this nasty bias problem.  And now that I've been using it for a while,  I know why, which is that it has a very high 
variance, which means when I call it multiple  times with just like samples with different  random seeds, I get very different values. 
And so I actually haven't  found this useful at all. 
So we left in the situation, which is, yeah,  we don't actually have a good unbiased metric. 
And I think that's the truth of  where we are, the best practices. 
And even if we did, all I would tell you is  like how similar distributions are to each 
other. It doesn't actually   tell you whether they look any good really. So that's why in pretty much all good papers,  
they have a section on human testing. But I've definitely found this useful  
for me for like comparing fashion images, which particularly like humans are good at looking at  
like faces that are reasonably high resolution and be like, Oh, that eye looks kind of weird,   but we're not looking good at looking at 28 by 28 fashion images. 
So it's particularly helpful for  stuff that our brains aren't good at.  So I basically wrapped this up into a class,  which I called ImageEval for evaluating images. 
And so what you're going to do is you're going  to pass in a pre-trained model for a classifier 
and your Dataloaders, which is the thing that  we're going to use to, to basically calculate 
the real images. So that's going to be,  
you know, the Dataloaders that were in this learn. So the real images. 
And so what it's going to do in this class  that again, I, this is just copying and pasting  the previous lines of code  and putting them into a class. 
This is going to be then something  that we call capture_preds on   to get our features for the real images. 
And then we can also calculate  the stats for the real images.  And so then we can call fid by calling _calc_fid,  which is the thing we already had passing 
in the stats for the real images and calculated  stats for the features from our samples, where 
the features, the thing that we've  seen before, we pass in our samples,   any random Y value is fine. 
So I just have a single tensor  there and call capture_preds.  So we can now create an ImageEval mod object  passing in our classifier, passing in our 
Dataloaders with the real data,  any other callbacks you want.  And if we call fid, it takes about a quarter of  a second and 33.9 is the FID for our samples. 
So something that I think, okay, then KID,  KID’s going to be a very different scale. 
It's only 0.05. So KIDs are generally much smaller than FIDs. 
So mainly going to be looking at FIDs. And so here's what happens if we call FID  
FID and KID plots
on sample 0 and then sample 50 and then sample 100 and so forth all the way up to 900. 
And then we also do samples, 975, 990, 999. And so you can see over time,  
our samples’ FID’s improved. So that's a good little test.  There's something curious about the fact  that they stopped improving about here. 
So that's interesting. I have not seen anybody plot this graph before.  I don't know if Johno or Tanishq, if you guys  have, I feel like it's something people should 
be looking at because it's really telling you:  is your sampling making consistent improvements. 
JOHNO: And to clarify, this is like the predicted  denoised sample at the different stages during  sampling, right? JEREMY: Yes, exactly. 
JOHNO: If I was to stop something now and just  go straight to the predicted X error, what would  the FID be? JEREMY: So  
I just want to check or check our samples. Yeah, we preset, we add the x0_hat at each time. 
Yep. Yep.  Exactly. Same for KID.  And I was hoping that they  would look the same and they do. 
So that's encouraging that KID and FID  are basically measuring the same thing.  And then something else that I haven't seen  people do, but I think it's very good idea 
is to take the FID of an actual batch of data.  Okay. And so that's tells us how good we could get. 
Now that's a bit unfair because I think the  different sizes, our data is 512, our sample 
was 256, but anyway, it's  a pretty huge difference. 
And then, yeah, the second thing that Tanishq  talked about, which I thought I'd actually  show is what does it take to  use, you know, to get a real FID? 
Real FID - The Inception network
There's the Inception network. So I didn't particularly feel like   re-elementing the Inception network. So I guess I'm cheating here. 
I'm just going to grab it from pytorch_fid. But there's absolutely no reason   to study the Inception networks. It's totally obsolete at this point. 
And as Tanishq mentioned, it wants 299 by  299 images, which actually you can just call 
resize input to have that done for you. It also expects three channel images. 
So what I did is I created a wrapper for an  Inceptionv3 model that when you call forward, 
it takes your batch and replicates  the channel three times. 
So that's basically creating a three channel  version of a black and white image just by 
replicating it three times. So with that wrapping, and again, this  
is good, like flexing of your PyTorch muscles, you know, try to make sure you can replicate this,  
that you can, yeah, get an Inception model working on your Fashion-MNIST samples. 
And yeah, then from there we can just  pass that to our image eval instead. 
And so on our samples, that gives us 63.8  and on a real batch of data, it gets 27.9. 
And like, I find this like a good sign that  this is much less effective than our real 
Fashion-MNIST classifier, because like that's  only a difference of a ratio of three or so. 
The fact that our FID for real data using  a real classifier was 6.6, I think that's 
pretty encouraging, you know. Yeah, so that is that. 
And we now have a FID, more specifically,  we now have an ImageEval class. 
Did you guys have any questions or  comments about that before we keep going?  TANISHQ: No, let's do it. JOHNO: I think again that  
pretty much every other FID you see  reported is going to be, you know,  set up for CIFAR-10 tiny 32 by 32 pixels resized  up to 299 and fed through Inception that was 
aimed on imaging, not CIFAR-10. So yeah, it's bearing in mind that once again,   this is a slightly weird metric and even things like the types of image, like the imagery sizing  
algorithms and PyTorch and TensorFlow might be slightly different.  Or you know, if you saved your images as JPEGs  and then reloaded them, your FID might be 
twice as bad. JEREMY: Yeah, it makes a big difference.  JOHNO: Yeah, exactly. So just to reiterate, what this will like,  
the takeaway from all of this that I get is that it's really useful.  Everything's the same, like using the same  backbone model, using the same approach, the 
same number of samples, then you  can compare it apples to apples. 
But yeah, for one set of experiments, a FID of  30 might be good because of the way everything's 
set up and for another, that might be terrible. So trying to compare to a paper or whatever is  
best. JEREMY: So I'm going to say...  TANISHQ: I guess maybe the approach is that like,  if you're doing your own experiments, you know, 
these sorts of metrics are good, but then  if you're going to compare to other models,  it's best to rely on human studies if you're  comparing to other models and that, yeah, 
I think that's kind of the sort of approach  or mindset that we should be having when it 
comes to this. JEREMY: Yeah, or both, you know.  But yeah, so we're going to see this  is going to be very useful for us. 
And we're just going to be using the same,  pretty much all the time, we're going to use  the same number of samples and we're going to  use the same Fashion-MNIST specific classifier. 
So the first thing I wanted to do was fix  our bug and to remind you, the bug was that 
Fixing (?) UNet feeding - DDPM_v3
we had, we were feeding into our UNet in DDPMv2  and the original DDPM images that were from 
zero to one. And yeah, that's wrong.  Nobody does that. Everybody feeds in images  
that are from minus one to one. So that's very easy to fix. 
You just… JOHNO: Jeremy, just to  ask, like, why, why is that a bug?  JEREMY: Why it's a bug?. I mean, it's like, everybody knows  
it's a bug because that's what everybody does. Like I've never seen anybody do anything   else and it's very easy to fix. So I fixed it by adding this to  
DDPMv2 and I reran it and it didn't work. It made it worse. 
And this was the start of, you know, a few  horrible days of pain because like when you, 
you know, fix a bug and it makes things worse,  that generally suggests there's some other 
bug somewhere else that somehow  has offset your first bug.  And so I had to go, you know, I basically  went back through every other notebook at 
every cell and I did find at least one bug  elsewhere, which is that we hadn't been shuffling 
our training sets the whole time. So I fixed that, but it's got   absolutely nothing to do with this. And I ended up going through everything  
from scratch three times, rerunning everything three times, checking every intermediate output   three times. So days of,  
you know, depressing and annoying  work and made no progress at all. 
At which point I then asked Johno's question  to myself more carefully and provided a less 
flippant response to myself, which was, well,  I don't know why everybody does this, actually. 
So I asked Tanishq and Johno and I was like,  and Pedro and I was like, have you guys seen  any math papers, whatever it's based  on this particular input range? 
And yeah, you guys are both like, no, I haven't. It's just, it's just what everybody does. 
So at that point it raised the  possibility that like, okay, maybe, maybe  
what everybody does is not the right thing to do.  And is there any reason to believe  it is the right thing to do? 
And given that it seemed like fixing  the bug made it worse, maybe not. 
And then, but then it's like, well, okay,  we, we are pretty confident from everything 
we've learned and discussed that having  centered data is better than uncentered data.  So having data that go from  zero to one clearly seems weird. 
So maybe the issue is not that we've changed  the center, but that we've scaled it down.  So rather than having a range  of two, it's got a range of one. 
So at that point, you know, I, I did  something very simple, which was, I  
did this, I subtracted 0.5.  So now rather than going from naught  to one, it goes from minus 0.5 to 0.5. 
And so the theory here then was, okay, if  our hypothesis is correct, which is that the 
negative one to one range has no foundational  reason for being, and we've accidentally hit 
on something, which is that a range of one is  better than a range of two, and this should  be better still, because this is a  range of one and it's centered properly. 
And so this is DDPM_v3   and I ran that and yes, it appeared to be better. And this is great because now I've got FID. 
I was able to run FID on DDPM_v2 and on  DDPM_v3 and it was dramatically, dramatically,  
dramatically better.  And in fact, I was running a lot, a lot of  other experiments at the time, which we'll 
talk about soon. And like all of my experiments are   totally falling apart when I fixed the bug. And once I did this, all the things that I  
thought, thought weren't working, suddenly started working.  So you know, this is often  the case, I guess, is that  
bugs can highlight, you know, accidental discoveries and the trick is always to be,  
you know, careful enough to recognize when that's happened.  You know, this is, some people might  remember the story, this is how the  
noble gases were discovered.  A chemistry experiment went wrong and left  behind some strange bubbles at the bottom 
of the test tube. And most people   would just be like, huh, oops, bubbles. But people who are careful enough actually went,  
no, there shouldn't be bubbles there. Let's test them carefully.  It's like, they don't react. Again, most people would be like, oh,  
that didn't work. The reaction failed.  But you know, if you're really careful, you'll  be like, oh, maybe the fact that they don't 
react is the interesting thing. So yes, being careful is a fair feature. 
JOHNO: When you say things like it didn't work  or it was worse, when you first showed us this  thing, I kind of said, oh, the images looked fine. The FID was slightly worse. 
But it was okay. And if you trained it longer,   it eventually got better. Mostly.  There were some things that sampling occasionally  went wrong, one image in a hundred or something 
like that. But it was like,   this isn't like everything completely fell apart. No, it's just the truth. 
The performance was slightly worse than expected. If you were doing the run and gun, try a bunch of  
things, it was like, oh, well, I just doubled my training time and set a few runs going   and looked at the Weights and Biases stats later and oh, that seems like it's better now. 
We just needed to train for longer. And we have infinite GPUs and lots of money.  You wouldn't notice this. So yeah, it wasn't like, you know, yeah,  
the fact that you picked up on it showed that you had this like deep intuition for where   it should be at this stage in training versus where it was, what the samples should look like. 
So there were any other fit as well to say  like, okay, I would have expected a FID of  nine and I'm getting 14. What's up, what's up here. 
And that was enough to start asking these  questions and we all jumped on all the other  to think, you know, where this came from. JEREMY: Yeah. 
I mean, definitely, I drive  people crazy that I work with.  I don't know why you guys aren't crazy yet,  but with this kind of like, no, I need to 
know exactly why, you know, this  is not exactly what we expected. 
But yeah, this is why to, to find the, you  know, when something's mysterious and weird, 
it means that there's something you didn't  understand and that's an opportunity to learn  something new. So that's what we did. 
And… so that was quite exciting because  yeah, going minus 0.5 to 0.5 made the FID  better still. 
And I was definitely in, yeah, I moved  from this frame of mind or from like,  
you had a whole total depression.  I was so mad. I still remember when I spoke to Johno,  
I was just so upset and, you know, and that I was suddenly like, Oh my gosh,   we're actually onto something. So, started experimenting more and you know,  
a bit more confidence at this point, I guess. And one thing I started looking at was our,  
Schedule experiments
our schedule, you know, I, you know, we'd always been copying and pasting this  
standard, again, set of stuff. And I started questioning everything.  Why is this a standard? Like, why are these numbers here? 
You know, and I don't see any particular  reason why those numbers were there. 
And I thought, well, we should  maybe experiment with them.  So to make it easier, I created a little  function that would return a schedule. 
Now you could create a new class for a schedule,  but something that's really cool is there's 
a thing in Python called SimpleNamespace,  which is a lot like a struct in C basically  lets you wrap up a little bunch of keys and  values as it lifts as if it's an object. 
So okay, this little SimpleNamespace, which  contains our alphas, our alpha bars and our 
sigmas for our normal betamax equals  0.02, linspace, this is what we always do. 
And then yeah, there's another paper which  mentions an alternative approach, which is 
cosine and… cosine schedule, which is where  you basically set alpha bar equal to t as 
a fraction of big T times pi  over two, cosine of that squared. 
And if you make that your alpha bar, you can  then basically reverse back out to calculate 
what alpha must have been. And so we can create a   schedule for this cosine schedule as well. And yeah, this cosine schedule is, I think,  
pretty recognized as being better than this linear schedule. 
And so I thought, okay, it'll be  interesting to look at how they compare.  And in fact, really all that  matters is the alpha bar. 
The alpha bar is, you know, the total  amount of noise that you're adding. 
So in DDPM, when we do noisify, you know,  it's alpha bar that we're actually using. 
JOHNO: The amount of the image and one  minus alpha bar, it's the amount of noise.  JEREMY: Exactly. Yeah. 
So I just printed those out, plotted those  for the normal linear schedule and this cosine 
schedule. And you can really see the linear schedule.  It really sucks badly. It's got a lot of  
time steps where it's basically about zero. And you know, that's something we can't really  
do anything with, you know, whereas the cosine schedule is really nice and smooth and there's not  
many steps which are nearly zero or nearly one.  So I thought, so I was kind of inclined to  try using the cosine schedule, but then I 
thought, well, it'd be easy enough to get  rid of this big flat bit by just decreasing 
betamax. That'd be another thing we can do.  So I tried, oh, sorry, first of all, I should  mention that the other thing that's really 
important is the slope of these  curves, because that's how much   things are stepping during the sampling process. 
And so here's the slope of the lin and the  cosine, and you can see the cosine slope. 
Really nice, right? You have this nice smooth curve,   whereas the linear is just a disaster. So yeah, if I change betamax to 0.01,  
that actually gets you nearly the same curve as the cosine. 
So I thought that was very interesting. It kind of made me think like, why on earth   does everybody always use 0.02 as the default? And so we actually talked to Robin, who was one  
of the two lead authors on the Stable Diffusion paper, and in fact we talked   about all of these things, and he said, oh yeah, we noticed not exactly this, but we experimented  
with everything, and we noticed that when we decreased betamax, we got better results.  And so actually Stable  Diffusion uses betamax of 0.012. 
I think that might be a little bit higher than  they should have picked, but it's certainly  a lot better than the normal default. So it's interesting talking to Robin to see  
all of these kinds of experiments and things that we tried out, they had been there  
as well and noticed the same things. JOHNO: The inputs range as well, they have this  
magical factor of 0.18, 0.02, whatever, they scale the latency by.  And if you ask why, they're like, oh yeah,  we wanted the latest to be roughly uniform 
range, whatever. But that's also like, that's   reducing the range of your inputs to reasonable. JEREMY: I think exactly,  
we independently discovered and they  independently discovered this idea.  Yeah, exactly. Yeah, exactly. 
So we'll be talking more about like what's  actually going on with that maybe next lesson. 
Anyway, so here's the curves as  well, they're also pretty close.  So at this point I was kind of thinking, well,  I'd like to like change as little as possible, 
so I'm going to keep using a linear schedule,  but I'm just going to change betamax to 0.01. 
For my next, you know, version of DDPM. So that's what I've got here,   linear schedule betamax 0.01. And so that I wouldn't really have to  
change any of my code, I'd end up just put those in the same variable names that I've always used.  So then noisify is exactly the  same as it always has been. 
So now I just repeat everything  that I've done before.  So now would I show a batch of data, I can  already see that there's, you know, more actually 
recognizable images, which  I think is very encouraging.  Previously they, like almost all of them had  been pure noise, which is not a good sign. 
So okay, so now I just train  it exactly the same as DDPM_v2. 
Train DDPM_v3 and testing with FID
And so I save this as Fashion DDPM_v3. Oh, and then the other things I've done here is,  
you know, this turned out to work pretty well.  I actually decided, let's keep going even further. So I actually doubled all of my channels  
from before. And I also increased the number of epochs   by three, because things are going so well, I was like, how well could they go? 
So we've got a bigger model trained  for longer, so it takes a few minutes. 
That's what the 25 here is the number of epochs. So samples exactly the same as it always has been. 
So create 512 samples. And here they are, and they definitely look to me,  
you know, great, like that. I'm not sure I could recognize whether  
these are real samples or generated samples. But luckily, we know we can test them. 
So we can load up our data_aug_2, delete  the last two layers, pass that to ImageVal, 
and get a FID for our samples. And it's 8.  And then I chose 512 for a reason,  because that's our batch size. 
So then I can compare that like with like  for the FID for the actual data at 6.6. 
So this is like hugely exciting to me, we've  got down to a FID that is nearly as good as 
real images. So I feel like this is,  
you know, in terms of image  quality for small, unconditional 
sampling, I feel like we're  done, you know, pretty much. 
And so at this point, I was like, okay, well,  can we make it faster, you know, at the same  quality. And I just wanted to experiment  
with a few things, like really obvious ideas. And in particular, I thought, we're calling this  
1,000 times, which means we're calling this 1,000 times, just running the model. 
And that's slow. And most of the time, you just move a tiny bit.  So the model is pretty much the same. It's, you know, the noise being  
predicted is pretty much the same. So I just did something really obvious, which   is I decided, let's only call the model every third time, you know, and maybe also just  
the last 50 to help it fine tune. I don't know if that's necessary,   other than that it's exactly the same. So now this is basically three times faster. 
And yeah, samples look basically the same. So the FID is 9.78 versus 8.1. 
And this is like within  the normal variance of FID.  So I don't know, like you'd have to run this  a few times or use bigger samples, but this 
is basically saying like, yeah, you probably  don't need to call the model a thousand times. 
I did something else slightly weird, which  is I basically said like, oh, let's create  a different like schedule for how often we  call the model, which is I created this thing 
called sample_at. So I basically said   when you're for the first few time  steps, just do it every 10 and then 
for the next 2, every 9 and then  the next 3, every 8, so forth.  And just for the last 100, do it every 1. So that makes it even faster. 
Samples look good. This is, you know,   it's definitely worse though now,  you know, but it's still not bad. 
So yeah, I kind of felt like,  all right, this is encouraging.  And this, this stuff before we fixed the minus  one to one thing was, they looked really bad, 
you know, um, that's why I was  thinking that my code is full of bugs. 
So at this point I'm thinking, okay, okay, okay.  We can create extremely high  quality samples using DDPM. 
Denoising Difussion Implicit Models - DDIM
What's the like, you know, best  paper out there for doing it faster. 
And the most popular paper  for doing it faster is DDIM. 
So I thought we might switch to this next. So we're now at the point where we're not  
actually going to retrain our model at all. If you noticed with these different sampling  
approaches, I didn't retrain the model at all, but just say, okay, we've got a model.  The model knows how to best  to make the noise in an image. 
How do we use that to call it multiple times  to denoise using iterative refinement as Johno 
calls it. And so DDIM is a, another way of doing that. 
So what we're going to do, I'm going to show  you how I built my own DDIM from scratch and 
I kind of cheated, which is, there's  already an existing one in diffusers. 
So I decided I will use that first,  make sure everything works and then  
I'll try and re-implement it from scratch myself.  So that's kind of like when there's an existing  thing that works, you know, that's what I 
like to do. And it's been really good to have my own   DDIM from scratch because now I can modify it, you know, and I've made it much more  
concise code than the diffusers version. So, now, we had created this class called a UNet,  
which asked the tuple of Xs through as individual parameters  
and returned to the dot sample. But not surprisingly, given that this comes  
from diffusers and we want to use the diffusers schedulers, the diffusers  
schedulers assume this has not happened. It wants the X, you know, as a tuple and it  
expects to find the thing called dot sample. So here's something crazy.  When we save this thing, this pickle, it doesn't  really know anything about the code, right? 
It just knows that it's from a class called UNet. So we can actually lie. 
We can say, oh yeah, that class called UNet it's actually the same as a UNet2DModel with no  
other changes and Python doesn't know or care, right?  So this, we can now load up this  model and it's going to use this UNet. 
Okay. So this is where it's useful to understand   how Python works behind the scenes, right? It's a very simple programming language. 
So we've now got a model which we've trained,  but it's not, it's just going to, you know, 
use the dot sample. And that means we can   use it directly with the diffusers schedulers. So we'll start by actually repeating what we  
already know how to do, which is use a DDPM scheduler.  So we have to tell it what beta we used to train. And so we can grab some random data. 
And so we could say, okay, we're  going to start at time step 999. 
So let's create a batch of data  and then predict the noise. 
And then this is the way that diffusers thing  works is you call scheduler.step and that's 
the thing which does those lines. That's the thing that calculates x_t given noise. 
So that's what scheduler.step does. That's why you pass in   x_t and the time step and the noise. And that's going to give you a new set. 
And so I ran that as usual first cell by cell  to make sure I understood how it all worked.  I then copied those cells and merged  them together and chucked them in a loop. 
So this is now going to go through all the  time steps, use a progress bar to see how  we're going, get the noise, call step and append. So this is just DDPM, but using diffusers and  
not surprisingly it gives us, you know, basically the same results as, you know,  
nice results, very nice results that we got from our own DDPM.  And so we can now use the same code we've  used before to create our image evaluator. 
And I decided, yeah, we're now going to  go right up to 2,048 images at a time. 
So it's now, this is the size I found it's  big enough that it's reasonably stable.  And so we're now down to 3.7 for our FID,  where else the data itself has a FID of 1.9. 
So again, showing that our DDPM is basically  very nearly unrecognizably different from 
real data using its distribution  of those activations.  So then we can switch to DDIM  by the same DDIM scheduler. 
And so with DDIM, you can say, I  don't want to do all thousand steps.  I just want to do 333 steps to every third. So that's basically a bit like, a bit like  
this sample skip of doing every third, but DDIM as we'll see, does it in a smarter way. 
And so here's exactly the same code basically  as before, but I put it into a little function. 
Okay.  So I can basically pass in my  model, the size, the scheduler. 
And then there's a parameter called eta,  which is basically how much noise to add. 
So just add all the noise. And so this is now going to  
take three times, this three times faster. And yeah, the FID's basically the same. 
That's encouraging. So I went down to 200 steps.  It's basically the same. 100 steps. And at this point, okay, the FID's getting worse. 
And then 50 steps. We're still. 25 steps. 
We're still... It's interesting, like when you get down   to 25 steps, like, well, what does it look like? And you can see that they're kind of like,  
they're too smooth. You know, they don't have interesting,  
you know, fabric swells so much or buckles or mongos or patterns as much, you know, as the,  
these ones, they've got a lot more texture to them.  So that's kind of what tends to happen. So you can still like get something out pretty  
fast, but that's, that's kind of how they suffer.  So okay. So how does DDIM work? 
How does DDIM works?
Well DDIM, it's nice. It's actually, in my opinion,   it makes things a lot easier than DDPM. So there's basically an equation from  
the paper, which Tanishq will explain shortly. But basically what you do is I've actually grabbed  
the sample function from here and I split it out into two bits. 
One bit is the bit that says, what are the  time steps, creates that random starting point, 
loops through, finds what my current a bar  is, gets the noise, and then, basically does 
the same as sched.step. It calls some function, right?  And then that's been pulled out. So this allows me to now create my own  
different steps. So I created a DDIM step and basically all  
I did was I took this equation and I turned it into code. 
Actually this, this one is the  second equation from the paper.  Now it's a bit confusing, which is  that the notation here is different. 
DDPM, what it calls alpha  bar, this paper calls alpha. 
So you've got to look out for that. So basically you'll see, I basically go,   I've got here x_t, x_t minus, okay. One minus alpha bar is we've  
created a call that beta bar. So beta bar dot square root times noise. 
This here is the, this is the neural net. So this here is the noise. 
Okay.  And here I've got my next x_t is, oh, sorry. Yes. 
Here's my a bar t one square root times this,  and you can see here, it says predicted X 
naught. Here's my predicted X naught plus   beta bar t one minus sigma squared square root. 
Again, here's noise. It's the same thing as here. 
Okay. And then   plus a bit of random noise, which we  only do if you're not at the last step. 
So yeah, so I can call that. So I just did it for,  
so I'd rather than saying a hundred  steps, I said, skip every skip 10  steps. So do 10 steps at a time. 
So it's basically going to be a hundred steps. And so you can see here actually is just happened  
to do a bit better for my a hundred steps. It's not bad at all. 
So yeah, I mean, this has been getting to  this point, it's been a bit of a lifesaver 
to be honest, because you know, I can now  run a two, you know, two, that batch of 
2,048 samples. I can sample them in under a minute   which doesn't feel painful. And so I'm now at a point  
where I can actually get a pretty  good measure of how I'm doing 
in a pretty reasonable amount of time. And I can, you know, easily compare it. 
And I got to admit, you know, the difference  between a FID of 5 and 8 and 11, I  can't necessarily tell the difference. So for fashion, I think FID is better than my eyes  
for this, as long as I use a consistent sample size. 
So yeah, Tanishq, did you want to talk a bit  about, you know, the ideas of why we do this 
or where it comes from or what the notation means? JOHNO: Can I say a little bit before we do that,  
Notation in Papers
which is just that what you have there, Jeremy, which is like a screenshot from the paper and then  
the code that is closest as possible tries to follow that.  Like the difference that makes for people is huge. Like I've got a little research team that  
I'm doing some contract work with. And the fact that like it's called   alpha in the DDIM paper and alpha bar elsewhere. And then in the code that they were copying and  
pasting from, it was called A and B for alpha and beta.  And it's like you can get things kind of  working by copying and pasting into things. 
And it's all just sort of kind of works, but  just spending that time to literally take  two screenshots from equation 14 and 16 from  the paper and put them in there and rewrite 
the code so that it, you know, with some comments  and things to say like, this is what this  is, this is that part from the equation. It's like, you know, the, the look of pain on  
their face when I said, Oh, by the way, did you notice that like,   it's called alpha there and alpha bar there? They're like, yes, how could they do that? 
You know, it's just like, you could just tell  how many hours have been spent, you know,  like grinding and saying what's wrong here. JEREMY: And doing notebooks. 
Yeah, and building this stuff in  notebooks is such a good idea.  Like we're doing with miniai because the,  you know, the next engineer to come along 
and work on that and see the equation  right there and you can add rows and stuff. 
So I think, you know, nbdev works particularly  well for this, this kind of development. 
Yeah. Thanks, Johno.  TANISHQ: Yeah, before, I talk about this,  I just wanted to briefly, in the context 
of all of these differing notations, I recently  created this meme, which I thought was, was 
relevant in terms of like each paper basically  has a different diffusion model notation. 
So it's just like this, but they all try to  come up with their own universal notation  and it's just, just keeps proliferating. JEREMY: Let's just agree we  
should all use APL. TANISHQ: Yes, exactly.  We need to implement diffusion  models in APLs somehow. 
DDIM paper
So yeah, the paper that we're, that, you know,  Jeremy had implemented was this Denoising 
Diffusion Implicit Model paper. And if you look at the paper again, you could  
see like the notation could be again, a little bit intimidating, but when we walk through it,  
we'll see, it's not too bad actually. So I'll just bring up, I guess, some of  
the important equations and also comparing and contrasting you know, DDPM and the notation  
of DDPM and the equations with DDIM. JEREMY: Not only is it not too bad,   I actually discovered it's making  life a lot, the DDIM notation 
and equations are a lot  easier to work with than DDPM.  So I found my life is better  since I discovered DDIM. 
TANISHQ: Yes, yes, I think a lot of  people prefer to use DDIM as well. 
So yeah, basically in both DDIM and  DDPM, we have this same sort of equation. 
This equation is exactly the same. This is telling us the predicted denoised image. 
So we predict our, but basically we  predict the, you can see my pointer, right? 
Just want to confirm. JEREMY: By the way, the little double-headed   arrow in the top right, does that, if you click that till you get more room  
for us to see what's going on. The double-headed arrow just above the, yeah.  TANISHQ: Yeah, sorry. Yeah. 
That’s much better. So we have our predicted noise. 
So our model is predicting the noise in the image. It is also passed in the time step,  
but this is just emitted. It basically kind of is given in the XT,   but our model also takes in the time step. And so it's predicting the noise in this  
XT, our noisy image. And we are trying to remove the noise.  That's what this whole term  here is, remove the noise. 
So because our noise that we're  predicting is unit variance noise. 
So we have to scale the variance of our noise  appropriately to remove it from our noisy 
image. So we have to scale the noise   and subtract it out of the original image. And that's how we would get our  
predicted denoised image. JEREMY: And I think we have   derived this one before by looking at the  equation for XT in the noisify function 
and rearranging it to solve for X nought. And that's what you get. 
TANISHQ: Yes, that's basically what this is. Yes, that's basically what this is.  So basically the idea is, okay, instead of,  yeah, noisifying it where we're starting out 
with X0 and some noise and get an XT, we're  doing the opposite where we have some noise 
and we have XT. So how can we get X0?  So that's what this equation is. So that's the predicted X0  
or our predicted clean image. And this equation is the same for both DDPM  
and DDIM, but these distributions are what's different between DDPM and DDIM. 
So we have these distribution, which tells us,  okay, so if we have XT, which is our current 
noisy image; and X0, which is our clean image,  can we find out what some sort of intermediate 
noisy image is in that process? And that's XT minus one.  So we have a distribution for that. And so that tells us how to get such an image. 
And so this is in the DDPM paper, they did  to divide some distribution and explain the 
math behind it. But yeah, basically they have some equation.  So you have, again, a Gaussian distribution  with some sort of mean and variance, but it's 
again some form of you have this  sort of interpolation between  
your original clean image and your noisy image, and that gives you your  
intermediate slightly less noisy image. So that's what this is giving. 
Given a clean image and a noisy image,  you're slightly less noisy image.  And so the sampling procedure that we do with  DDPM basically is predict the X0 and then 
plug it into this distribution to give  you your slightly less noisy image. 
So maybe it's worth drawing that out. So if we had, let's say, some sort of, I don't  
know, I'm just making some sort of, I don't know, maybe a lot of some sort of better,   yeah, some sort of. So then in this case,  
I'm showing a one dimensional example. Let's say you have some sort of a point.  So it's kind of a one dimensional example  that's still in the sort of 2D space. 
But let's say you have some, any point on  this represents an actual image that you want 
to sample from, right? So this is where your   distribution of actual images would lie. And you want to estimate this. 
So this sort of algorithm that we've been  seeing here says that, okay, if we take some 
random point, this is some random  point that we choose when we start out.  And what we did is we learned this function,  the score function to take us to this manifold, 
but it's only going to be accurate in some space.  So it's going to be accurate. It would be accurate in some area. 
So we get an estimate of the score function  and it tells us the direction to move in.  And it's going to give us the direction  to predict our denoised image, right? 
So basically, let's say your score function  is actually in reality some curve, okay? 
So it's in reality some curve that  points to your, oops, it points here.  So that's your score function. And you know the value here. 
JEREMY: Just to clarify, the score function basically   means your gradient, yeah? TANISHQ: Yes, yes.  It's a gradient. So we're again doing some form of, in this case,  
I guess you would say gradient ascent, because you're not really minimizing   the score, you're maximizing it. You want, so you're maximizing the likelihood  
of that data point being an actual data point. You want to go towards it.  So you're doing the sort  of gradient ascent process. 
So you're following the gradient to get to that. So when we estimate epsilon, theta, and predict  
our noise, what we're doing is we're getting the score value here.  And then so we can follow that  and we follow it to some point. 
And being kind of exaggerating here, but  this point will now represent our X0 hat. 
So yeah, our X0 hat. And in reality, maybe that's not  
going to be some point that is an actual point. It wouldn't be next to the distribution.  So it's not going to be a very good estimate  of a clean image at the beginning, but we 
only had that estimate at the beginning at  this point, and we have to follow it all the  way to some place. So this is where we follow it to. 
And then we want to find  some sort of X t minus one. 
So that's what our next point is. And so that's what our second   distribution tells us. And it basically takes  
us all the way back to maybe some point here. And now we can re-estimate the score function or  
our gradient over there, do this prediction of noise.  And it may be more accurate of a score function. And maybe we go somewhere here, and then we  
re-estimate and get another point, and then we follow it.  And so that's kind of this iterative process  where we're trying to follow the score function 
to your own point. And in order to do so, we first have to estimate   our X0 hat, and then basically add back some noise and to get a little bit, get a new estimate  
and keep following and add back a little bit more noise and keep estimating.  So that's what we're doing  here in these two steps. 
We have our X0 hat, and then  we have this distribution.  And that's how we do it with regular DDPM. And I think that's maybe where the sort of  
breaking it up in two steps is a little bit clearer.  And I don't think the DDPM paper really  clarifies that or really talks about it too much. 
But the DDIM paper also really hammers that  point home, I think, and especially in their 
update equation. So that's the DDPM, but then with DDIM... 
Okay, go ahead. JOHNO: DDPM, just the one thing is that you   look at your prediction, use that to make a step, but you also add back some additional noise  
that's always fixed. TANISHQ: Right.  JOHNO: DDPM there's no parameter to control  how much extra noise you add back at each step. 
TANISHQ: Right, exactly. So when you're... 
Let's see here. So yeah, basically you won't   be exactly at this point. You could be...  You're in that general vicinity. Adding that noise also helps with... 
You don't want to fall into specific modes  where it's like, oh, this is the most likely 
data point. You want to add some noise where   you can explore other data points as well. So yeah, the noise also can help. 
That's something you really  can't control with DDPM.  And that's something that DDIM explores a  little bit further, is in terms of the noise 
and even trying to get rid of  the noise altogether in DDIM.  So with the DDIM paper, the main  difference is literally this one equation. 
That's all really it is in terms of changing  this distribution where you predict the less 
noisy equation. The less noisy...  Sorry, the less noisy image. 
And basically, as you can see, you have this  additional parameter now, which is sigma. 
And the sigma controls how much noise, like  we were just mentioning, is going to be part 
of this process. And you can actually, for example,   if you want, you could set sigma to zero. And then you can see here, now you have  
a variance that would be zero. And so this becomes a completely   deterministic process. So if you want, this could  
be completely deterministic. So that's one aspect of it. 
And then, yeah, so the other aspect... So one of the reasons it's called DDIM is just  
not DDPM because it's not probabilistic anymore.  It can be made deterministic. So the name was changed for that reason. 
But the other thing is you would think that  you've changed the model altogether with a 
new distribution altogether. And so you think, oh, wouldn't you   have to trade a different model for this purpose? But it turns out the math works out where the same  
model objective would work well with this distribution as well. 
In fact, I think that's what they were studying  out from the very beginning is what kind of  other models can we get with the same objective? And so this is what they're able to do is you  
can have this new parameter that you can introduce, in this case, kind of controlling  
the stochasticity of the model. And you can still use the same  
exact trained model that you had. So what this means is that this actually is  
just a new sampling algorithm and not anything new with the training itself.  And this is just, yeah, just like we talked  about, a new way of sampling the model. 
And then, so yeah, this is how, given now  this equation, then you can rewrite your X t 
minus one term. And again, we're doing the same sort of   thing where we split it up into predicting the X0 and then adding back to go back to your Xt. 
And also if you need to add a little bit of  noise back in, like Jonathan was saying, you 
can do so, you have this extra term  here and the sigma controls that term. 
And again, like we said, you have to be, again,  looking at the DIM equation versus the DDPM 
equation. You have to be careful of the alphas here are   referring to alpha bars in the DDPM equation. So that's the other caveat. 
So yeah, and you have this sigma t set to  this particular value will give you back DDPM. 
So sometimes instead they will write basically  Jeremy mentioned this sort of, I guess, eta, 
which is equal to basically, yeah, so it's  just basically eta is, sigma is equal to eta 
times this coefficient. So sorry, let me just go back. 
And so basically, yeah, in reality  you take, I'm not wearing white. 
You have the eta here. So it's like, yeah, this is where eta would go.  So if it's one, it becomes regular DDPM benefits  zero, of course that's a determinative case. 
So this is the eta that all these APIs and  in the code that we have, also the code that 
Jeremy was showing, they have eta equals to  one, which they say is corresponding to regular 
DDPM.  This is actually where the  eta would go in the equation. 
So finally the not- JOHNO: It's like, you could pass in sigma, right?  Like if you weren't trying to match it in  previous papers, you could just, oh, well, 
we have this parameter sigma that  controls the amount of noise.  So that's just taken sigma scale as an argument. And for convenience, they said, let's create  
this new thing, eta, where zero means sigma is equal to zero, which if you look at the  
equation that works, one means we match the DDPM, the amount of noise  
that's in the familiar DDPM. And so then that gives you a nice slice.  You could say eta equals two  or eta equals 0.7 or whatever. 
But it's got a meaningful unit of one equals  the same as this previous reference were. 
JEREMY: Well, it's also convenient because it's  sigma t, which is to say different time steps,  
unless you choose eta equals zero, in which case   it doesn't matter, different time steps probably want different amounts of noise. 
And so here's a reasonable  way of scaling that noise. 
TANISHQ: Then the last thing of importance, which  is of course, one of the reasons that we were 
exploring this in the first place, is  to be able to do this rapid sampling. 
The basic idea here is that you can define  a similar distribution where again, the math 
works out similarly, where now you have, let's  say you have some subset of diffusion steps. 
So in this case, it uses tau variable. So for example,  
let's say subset of diffusion steps. So if it's like 10 diffusion steps, then tau   one would just be zero, then tau two would be 10. 
You just keep going all the way up to  say 1,000, but you've only got the...  Sorry, tau two would be 100, and  then you go all the way up to 1,000. 
And so you'd get 10 diffusion steps. So that's what they're referring to  
when they have this tau variable here. And so you can do these sorts of similar  
equation and similar derivation  to show that this distribution 
here again, meets the objective  that you use for training. 
And you can now use this for a faster sampling,  where basically all you have to do is you 
have to just select the appropriate alpha bar. And sorry, this one I've written out. 
So this one, actually the alpha bar is the  regular alpha bar that we've talked about.  But basically, sorry, it's a little bit confusing  switching between different notations, but 
basically you have this distribution and then  you just have to select the appropriate alpha  bars and it follows the same in terms of  you have appropriate sampling process. 
So yeah, and I guess it makes it a lot simpler  in terms of doing this accelerated sampling. 
Yeah, I guess with any other note, maybe other  comments that maybe you guys had or was this... 
JEREMY: Well, the key for  me is that in this equation,  
we just have one, we only need one parameter, which is the alpha bar or alpha depending which  
notation is and everything else you calculated from that.  And so we don't have the  
what DDPM calls the alpha or beta anymore. And that's more convenient for doing this kind of  
smaller number of steps because we can just jump straight  
from time step to alpha bar. And we can also then, it's particularly convenient  
with the cosine schedule because you can calculate the inverse of the cosine schedule function,  
which means you can also go from an alpha bar to a T.  So it's really easy to say like, Oh, what  would alpha bar be 10 time steps previously 
to this one? You know,   it's just, you could just call a function. We don't need, yeah, we don't need anything else. 
And so actually the original cosine schedule  paper has to fuss around with various like 
kind of epsilon style small numbers that they  add to things to avoid getting weird numerical 
problems. And so, yeah, when we only deal with alpha bar,   all that stuff also goes away. So yeah, so looking, if you're looking  
at the DDIM code, you know, it's simpler code with less parameters than our DDPM code. 
And of course it's dramatically faster and  it's also more flexible because we've got  this eta thing we can play with. TANISHQ: Yes. 
Yeah. And that's the other thing is like this   idea of like, yeah, controlling stochasticity. I think that's something  
that's interesting to explore. And we've been exploring that a bit now   and I think we'll continue to explore that in terms of deterministic versus stochasticity. 
So yeah. JEREMY: So it's worth talking about just the   sigma in the middle equation you've got there. So you've got the sigma t, eta t, adding the  
random noise and intuitively it makes sense that if you're adding random noise there,  
you would need to have less. You want to move less back towards  
Xt, which is your noisy image. So that's why, and you know, you've got the   one minus alpha t minus one minus sigma squared, and then you're taking the square root of that. 
So basically that's just sigma,  the square root of the squared.  So you're subtracting sigma t from the direction  pointing to Xt and adding it to the random 
noise or vice versa. So everything's there for a reason,  
you know. TANISHQ: Yes. JEREMY: And the predicted X zero,   that entire equation we've derived previously. TANISHQ: And it remains the same in pretty  
much any diffusion model methodology. JEREMY: Well, as long as you're using,   we'll be talking about actually  some places where it's going 
to change probably next week. TANISHQ: Well, yeah, I guess… JEREMY: That's   another thing where you're predicting noise. Yes.  TANISHQ: Yes. Yes. 
If you're predicting the noise, yes, there'll be. JEREMY: Okay.  TANISHQ: Yeah. JEREMY: So,  
Wrapping up
so I think, you know, we will, we'll probably,  yeah, let's wrap it up here so that we leave 
ourselves plenty of time to cover the kind  of new research directions next lesson more  in more detail. I just wanted to mention like in terms  
of where we're at, just like we hit a kind of like, okay, we can really predict  
classes for Fashion-MNIST a few weeks ago where I think we're there now and like we can do Stable  
Diffusion, sampling and UNets, except for the UNet architecture for unconditional generation. 
Now we basically can do Fashion-MNIST almost. So it's unrecognizably different to the real  
samples and DDIM is the scheduler that the original Stable Diffusion paper used. 
So yeah, you know, we're actually about to  go beyond Stable Diffusion for our sampling 
and UNet trading now. So I think we've, yeah, definitely meeting our  
stretch goals so far and all from scratch with Weights and Biases, experiment logging. 
And, you know, if you wanted to have fun,  there's no reason you couldn't like have a 
little callback that instead logs  things into a SQLite database.  And then you could write a little front end  to show your experiments, you know, that'd 
be fun as well. JOHNO: Yeah.  I mean, you could do also send you a text  message when the loss gets good enough. 
Yeah. JEREMY: All right.  Well, thanks guys. That was really fun.  JOHNO: Thanks everybody. JEREMY: All right. 
Bye. Okay.  TANISHQ: Talk to you later then. Bye bye.

---

# 22

JEREMY: All right, hi gang, and   here we are in Lesson 21, joined by the  legends themselves, Johno and Tanishq. Hello.
TANISHQ: Hello. JEREMY: And today you'll be shocked to hear  that we are going to look at a Jupyter Notebook.  
Amazing, right? We're going to look at notebook  22. This is a pretty quick, just, you know,  
Cosine Schedule (22_cosine)
improvement, fairly simple improvement to our  DDPM - DDIM implementation for Fashion-MNIST.
And this is all the same so far, but what I've  done is I've made some, one quite significant  
change and some of the changes we'll be making  today are all about making life simpler.  
And they're kind of reflecting the way the  papers have been taking things. And it's  
interesting to see how the papers have not only  made things better, they've made things simpler.
And so one of the   things that I've noticed in recent papers is  that there's no longer a concept of n steps,  
which is something we've always had before and  always bothered me a bit, this capital T thing.  
You know, this t/T, it's basically saying this  is time step number, say 500 out of 1,000,  
so it's time step 0.5. Why don't I just call  it 0.5? And the answer is, well, we can. So  
we talked last time about the cosine scheduler.  We didn't end up using it because I came up with  
an idea which was, you know, simpler and nearly  the same, which is just to change our betamax.
But in this next notebook, let's use the cosine  scheduler, but let's try to get rid of the  
n_steps thing and the capital T thing. So here is  abar again. And now I've got rid of the capital T.  
So now I'm going to assume that your time step  is between 0 and 1. And it basically represents  
what percentage of the way through  the diffusion process are you.   So 0 would be all noise and 1 would  be all, no sorry, other way around.
0 would be all clean and 1 would be all noise.  So how far through the forward diffusion process.  
So other than that, this is exactly the  same equation we've already seen. And I   realized something else, which is kind of fun,  which is you can take the inverse of that.
So you can calculate t. So we would  basically first take the square root  
and we would then take the inverse cos and we  would then divide by 2 over pi or times pi over  
2. So we can both, so it's interesting now, we  don't, the alpha bar is not something we look up  
in a list, it's something we calculate with a  function from a float. And so yeah, interestingly,  
that means we can also calculate t from an alpha  bar. So noisify has changed a little. So now when  
we get the alpha bar for our time step, we don't  look it up, we just call it, call the function.  
And now the time step is a random float  between 0 and 1. Actually between 0 and 0.999,  
which actually I'm sure there's a function I  could have chosen to do a float in this range,  
but I just clapped it because I was  lazy, couldn't be bothered hooking it up.   Other than that noisify is exactly the  same. Right, so we're still returning the  
xt, the time step, which is now a float, and  the noise. That's the thing we're going to  
try and predict, dependent variable, this  tuple there as our inputs to the model.
All right, so here is what that looks like.  So now when we look at our input to our UNet  
training process, you can see, you know, we've  got a t of 0.05, so 5% of the way through the  
forward diffusion process, it looks like  this, and 65% through it looks like this.
TANISHQ: So now the time step, and  basically the process is more of  
a kind of a continuous time step  and a continuous process, rather,   before we were having these discrete time  steps, here we could, it's just any random  
value, it could be between 0 and 1.  I think, yeah, that's also something. JEREMY: Yeah, which I found is more convenient,  you know, to have a function to call.  
Yeah, I find this life a little bit easier. So  the model's the same, the callbacks are the same,  
the fitting process is the same. And so something  which is kind of fun is that we could now,  
we do now, create a little denoise function,  so we can take, you know, this batch of data  
that we generated, the noisified data, so  here it is again, and we can denoise it.  
So we know the t for each element, obviously,  so remember t is different for each element now.
Sampling
And we can therefore calculate  the alpha bar for each element,   and then we can just undo the noisification to  get the denoised version. And so if we do that,  
there's what we get. And so this is great, right?  It shows you what actually happens when we run a  
single step of the model on varyingly partially  noised images. And this is something you don't  
see very often, because I guess not many people  are working in these kind of interactive notebook   environments where it's really easy to do this  kind of thing, but I think this is really helpful  
to get a sense of like, okay, if you're 25% of  the way through the forward diffusion process,  
this is what it looks like when you undo that.   If you're 95% of the way through it, this is what  happens when you undo that. So you can see here,  
it's basically like, oh, I don't really know what  the hell's going on, so at least a noisy mess.
Yeah, I guess my feeling from looking  at this is I'm impressed, you know,  
like this 45% noise thing, it looks all  noise to me. It's found the long-sleeved top.  
And yeah, it's actually pretty close to the real  one. I looked it up, or you might see it later,   it's a little bit more of a pattern here, but it  even gives a sense of the pattern. So it shows you  
how impressive this is. So this is 35%, you can  kind of see there's a shoe there, but it's really  
picked up the shoe nicely. So it's, these are  very impressive models in one step, in my opinion.  
So okay, so sampling is basically the  same, except now, rather than starting with  
using the range function to create a time  steps, we use linspace to create our time steps. So our time steps start at, you know,  if we did 1,000, it would be 0.999,  
and they end at 0, and then they're just  linearly spaced with this number of steps.  
So other than that, you know, abar we now  
calculate, and the next abar is going to be  whatever the current step is minus one over steps.  
So if you're doing 100 steps, then you'd be minus  0.01. So this is just stepping through linearly.  
And yeah, that's actually it for changes.  So if we just do DDIM for 100 steps,  
you know, that works really well, we  get a FID of three, which is actually  
quite a bit better than we had on  100 steps for our previous DDIM. So this definitely seems like a good  
sampling approach. And I know Johno is  going to talk a bit more shortly about,  
you know, some of the things that can make  better sampling approaches. But yeah, definitely,  
we can see it making a difference here. Did you   guys have anything you wanted to  say about this before we move on?
Summary / Notation
JOHNO: No, but it is a nice transition towards  some of the other things we'll be looking at to   start thinking about how do we frame this. And  it's also good, like the idea… So the original  
DDPM paper has a thousand time-states and a lot  of people follow that. But the idea that you don't  
have to be bound to that, and maybe it is worth  breaking that convention. I know Tanishq made   that meme about, you know, this 15 computing  different standards for notation. But yeah,  
sometimes it's helpful to reframe it, okay, time  goes from zero to one, that can simplify some   things. It may complicate others, but yeah, it's  nice to think how you can reframe stuff sometimes.
JEREMY: Yeah, and in fact, where we will  head today, by the time we get to notebook   23, we will see, you know, even simpler  notation. And yes, simpler notation generally  
comes. I think what happens is over time,  people understand better what's the essence  
of the problem and the approach, and  then that gets reflected in the notation.  
So, okay, so the next one I  wanted to share is something which   is an idea we've been working on for a  while, and it's some new research. So partly,  
Predicting the noise level of noisy Fashion MNIST images
I guess this is an interesting  insight into how we do research.
This is 22_noise-spread. And the basic idea of  this was, well, actually, I'm going to take you  
through it to see what the basic idea is. So  what I'm going to do is I'm going to create,   okay, so Fashion-MNIST as before, but  I'm going to create a different kind  
of model. I'm not going to create  a model that predicts the noise,   given the noised image and t. Instead, I'm  going to try to create a model which predicts t,  
given the noised image. So why did I want to do  that? Well, partly, well, entirely, because I was  
curious. I felt like when I looked at something  like this, I thought it was pretty obvious,  
roughly, how much noise each image had.  And so I thought, why are we passing noise,  
when we call the model, why are we passing in  the noise image and the amount of noise or the t,  
given that I would have thought the model  could figure out how much noise there is. So I wanted to check my contention, which is  that the model could figure out how much noise  
there is. So I thought, okay, well, let's create a  model that would try and figure out how much noise   here is. So I created a different noisify now,  and this noisify grabs an alpha bar t randomly.  
And it's just a random number between 0 and 1,  you know, you want one per item in the batch.
And so then after just randomly grabbing an alpha  bar t, we then noisify in the usual way. But now  
our independent variable is the noise image,  and the dependent variable is alpha bar t. And  
so we're going to try to create a model that  can predict alpha bar t, given a noise image.  
Okay, so everything else is the same  as usual. And so we can see an example… JOHNO: …you've got alpha  bar t dot squeeze dot logit.
Why .logit() when predicting alpha bar t
JEREMY: Oh yeah, that's true. So the  alpha bar t goes between naught and one.  
So we've got a choice. Like I mean, we don't have  to do anything, but you know, normally if you've  
got say between zero and one, you might consider  putting a sigmoid at the end of your model.  
But I felt like the difference between 0.999 and  0.99 is very significant, you know. So if we do  
logit, then we don't need the sigmoid at the end  anymore. It'll naturally cover the full range of  
kind of, you know, it'll be centered at zero,  it'll cover all the normal kind of range of  
numbers, and it also will treat equal ratios as  equally important at both ends of the spectrum.  
So that was my hypothesis was that using logit  would be better. I did test it and it was actually  
very dramatically better. So without this logit  here, my model didn't work well at all. And so  
this is like an example of where thinking about  these details is really important. Because if  
I hadn't have done this, then I would have come  away from this bit of research thinking like, oh,   I was wrong. We can't predict noise, noise amount.  Yeah, so thanks for pointing that out, Johno.  
Yeah, so that's why in this example of a  mini-batch, you can see that the numbers   can be negative or positive. So 0 would represent  noise, the alpha bar of 0.5. So here 3.05 is not  
very noised at all, where else, negative one  is pretty noisy. So the idea is that, yeah,  
given this image, you would have to try to predict  3.05. So one thing I was kind of curious about is,  
like, it's always useful to know is like, what's  the baseline? Like, what counts as good? You know,   because often people will say to me like,  oh, I created a model and the MSE was 2.6.  
Random baseline
I'll be like, well, is that good? Well,  it's the best I can do, but is it good? Or is it better than random? Or is it  better than predicting the average?  
So in this case, I was just like, okay, well, what  if we just predicted, actually, this is slightly  
out of date, I should have said 0 here, rather  than 0.5, but nevermind close enough. So this is  
before I did the logit thing. So I basically  was looking at like, what's the, you know,  
loss if you just always predicted a constant,  which as I said, I should have put zero here,  
haven't updated it. And so it's like, oh, that  would give you a loss of 3.5. Or another way  
to do it is you could just, just put MSE here  and then look at the MSE loss between 0.5 and  
your various, just a single mini batch, which we,  yeah, mini batch of alpha bar t’s, logits. Yeah,  
so, you know, we wanted to get some, you know,  if we're getting something that's about three,   then we basically haven't done any better  than random. And so this case, this model,  
it doesn't actually have anything  to learn. It always returns the   same thing. So we can just call fit with  train equals false just to find the loss.
So these are just a couple of ways of getting  quickly finding a loss for a baseline naive model.  
One thing that thankfully PyTorch will warn you  about is if you try to use MSE and your inputs and  
mse_loss why .flatten()
targets have different shapes, it will broadcast  and give you probably not the result you would  
expect, and it will give you a warning. So one way  to avoid that is just to use dot flatten on each.  
So this kind of flattened MSE is useful to avoid  both, avoid the warning and also avoid getting  
weird errors or weird, sorry, weird results.  So we use that for our loss. So the models, the  
model that we always use, so it's kind of nice.  We just use our same old model. Nothing changes,  
even though we're doing something  totally different. Oh, well, okay,   that's not quite true. One difference is  that our output, we just have one output now,  
Model & results
because this is now a regression model.  It's just trying to predict a single number. And so our Learner now uses MSE as the loss.  Everything else is the same as usual. So we can  
go ahead and trade it and you can see, okay,  the loss is already much better than three,   so we're definitely learning something and we end  up with a 0.075 mean squared error. That's pretty  
good considering, you know, there's a pretty wide  range of numbers we're trying to predict here.
So I've got to save that as noise prediction  on sigma. So save that model. And so we  
can take a look at how it's doing by  grabbing our one batch of noise images,  
putting it through our tmodel. Actually, it's  really an alpha bar model, but never mind,  
call it a tmodel. And then we can take a look  to see what it's predicted for each one.And  
we can compare it to the actual for each  one. And so you can see here, it said, oh,   I think this is about 0.91. And actually it is  0.91. So now here it looks like about 0.36. And  
yeah, it is actually 0.36. So you know, you  can see overall 0.72, it's actually 0.72.  
Or it's actually right, this one's 0.02 off. But  yeah, my hypothesis was correct, which is that we,  
Why are we trying to predict the noise level?
you know, we can predict the thing that  we were putting in manually as input. So  
there's a couple of reasons I was interested in  checking this out. The first was just like, well,  
yeah, wouldn't it be simpler if we  weren't passing in the t each time? You know, why not pass in the t each time? But  it also felt like it would open up a wider range  
of kind of how we can do sampling. The idea of  doing sampling by like precisely controlling  
the amount of noise that you try to remove each  time, and then assuming you can remove exactly  
that amount of noise each time feels limited to  me. So I want to try to remove this constraint.
So having built this model, I thought, okay, well,  you know, which is basically like, okay, I think  
we don't need to pass t in. Let's try it. So what  I then did is I replicated the 22_cosine notebook,  
I just copied it, pasted it in here. But I made  a couple of changes. The first is that noisify  
Training diffusion without t - first attempt
doesn't return t anymore. So there's no  way to cheat. We don't know what t is.  
And so that means that the UNet now doesn't  have t, so it's actually going to pass 0  
every time. So it has no ability to learn   from t because it doesn't get t. So it  doesn't really matter what we pass in.
We could have changed the UNet to like remove  the conditioning on t. But for research,  
this is just as good, you know, for finding  out. And it's good to be lazy when doing   research. There's no point doing something  a fancy way when you can do it a quick and  
easy way before you even know if it's going  to work. So yeah, that's the only change.  
So we can then train the model and we can  check the loss. So the loss here is 0.034.  
And previously it was 0.033. So  interestingly, you know, maybe  
it's a tiny bit worse at that,  you know, but it's very close.
Okay, so we'll save that  model. And then for sampling,  
I've got exactly the same DDIM step as usual.  And my sampling is exactly the same as usual,  
except now when I call the  model, I have no t to pass in.  
So we just pass in this. I mean, I still  know t because I'm still using the usual  
sampling approach, but I'm not passing it  to the model. And yeah, we can sample and  
what happens is actually pretty garbage.  22 is our FID. And as you can see here,  
you know, some of the images are still really  noisy. So I totally failed. And so that's  
always a little discouraging when you think  something's going to work and it doesn't. But my reaction to that is like, if I think  something's going to work and it doesn't,  
is to think, well, I'm just going to have to do a  better job of it. You know, it ought to work. So  
I tried something different, which is I thought  like, okay, since we're not passing in the t,  
Why it isn’t working?
then we're basically saying like, how much noise  should you be removing? It doesn't know exactly.  
So it might remove a little bit more noise that  we want or a little bit less noise than we want.   And we know from the, you know, testing we did,  that sometimes it's out by like this case 0.02.
And I guess if you're out consistently,  sometimes it's, yeah, got to end up not  
removing all the noise. So the change I made  was to the DDIM step, which is here. And…  
let me just copy this and get rid of the, I'm into  that section just to make it a bit easier to read.
Okay. So the DDIM step, this is the normal  DDIM step. Okay. And so step one is the  
same. So don't worry about that. Cause it's  the same as we've seen before. But what I   did was I actually used my tmodel. So I  passed the noised image into my tmodel,  
which is actually an alpha bar model  to get the predicted alpha bar.   And this is, remember, the predicted alpha bar  for each image, because we know from here that  
sometimes, so sometimes it did a pretty  good job, right? But sometimes it didn't. So I felt like, okay, we need a  predicted alpha bar for each image.  
What I then discovered is, sometimes that could  be like really too low, right? So what I wanted  
to make sure of is it wasn't too crazy. So  I then found the median for a mini batch of  
all the predicted alpha bars, and I clamped  it to not be too far away from the median.  
And so then what I did when I did my  x_0_hat is rather than using alpha bar t,  
I used the estimated alpha bar t for each  image, clamped to be not too far away from  
the median. And so this way it was updating it  based on the amount of noise that actually seems  
to be left behind rather than the assumed  amount of noise that should be left behind.  
You know, if we assume it's  removed the correct amount. And then everything else is  the same. So when I did that,  
so whoa, made all the difference. And here it is.  They are beautiful pieces of clothing. So 3.88  
versus 3.2. That's possibly close enough.  Like I'd have to run it a few times,  
you know, my guess is maybe it's a  tiny bit worse. But it's pretty close.  
But like this definitely gives  me some encouragement that,  
you know, even though this is like something  I just did in a couple of days, where else   they're kind of the with t approaches have been  developed since 2015 and we're now in 2023.
I, you know, I would expect it's quite likely  that these kind of like no, no t approaches could  
eventually surpass the t based approaches.  
And like one thing that definitely makes me  think there's room to improve is if I plot  
the FID or the KID for each sample  during the reverse diffusion process,   it actually gets worse for a while. I'm  like, okay, well that's, that's a bad sign.
I have no idea when that's  happening, but it's a sign that,   you know, if we could improve each step, then  one would assume we could get better than 3.8. So  
yeah, Tanishkq, John, I don't have any thoughts  about that or questions or comments or…
Debugging (summary)
JOHNO: Maybe to just like to highlight the  research process a little bit, it wasn't like this  
linear thing of like, oh, here's this issue not  performing as well as we thought, oh, here's the   fix. We just kept this. You know, this was like  multiple days of like discussing and like Jeremy  
saying like, you know, I'm tying my hair out. You  guys have any ideas? And, oh, what about this? And, oh, and they're just in the DDIM paper,  they do this clamping, maybe that'll help,  
you know? So there's a lot of back  and forth and also a lot of like,   you saw the code that was commented out there,  print xt.min, xt.max, alpha bar, pred, you know,  
just like seeing, oh, okay. You know, my  average prediction is about what I would expect,   but sometimes the middle of the max goes, you  know, 2, 3, 8, 16, 150, 212 million, infinity,  
you know, maybe like one or two little  baddies that would just skyrocket out.  
Yeah. And so that kind of like debugging  and exploring and printing the results. JEREMY: And actually our initial discussions  about this idea, I kind of said to you guys  
before Lesson 1 of Part 2, I said, like, it  feels to me like we shouldn't need the t thing.  
And so it's actually been like bumbling  away in the background for months. Yeah.
JOHNO: And I guess, I mean, we should also  mention, we have tried this, like a friend of ours  trained a no-t version of stable diffusion for  us. And we did the same sort of thing. I trained  
a pretty bad t predictor and it sort of generates  samples. So we're not like focusing on that large  
scale stuff yet, but it is fun to like, every  now and again, got this idea from Fashio-MNIST,   we are trying these out on some bigger models  and seeing, okay, this does seem like maybe  
it'll work. And so down the line that future  plan is to say, let's actually, you know,   spend the time train a proper model and see,  yeah, see how well that does. If it's interesting.
JEREMY: You say a friend of ours, we can be more  specific. It's Robert, one of the two lead authors   of the stable diffusion paper who yeah, actually  has been fine tuning a real stable diffusion model  
which is without t and it's looking super  encouraging. So yeah, that'll be fun to play with,  
with this new, you know, we'll have to train  a t predictor for that. See how it looks. Yep.
All right. So I guess the other area we've been  talking about kind of doing some research on is  
Bug in ddpm - paper that cast some light on the issue
this weird thing that came up over the last two  weeks where our bug in the DDPM implementation  
where we accidentally weren't doing it  from minus 1 to 1 for the input range,  
it turned out that actually being from minus  one to one wasn't a very good idea anyway.  
And so we ended up centering it as  being from minus 0.5 to 0.5. And Johno  
and Tanishq have managed to actually  find a paper. Well I say find a paper.
A paper has come out in the last 24 hours   which has coincidentally cast some light on this  and has also cited a paper that we weren't aware  
of which was not released in the last 24 hours. So  Johno, are you going to tell us a bit about that?
JOHNO: Yeah, sure I can do that. So it's  funny, this was such perfect timing because  
I actually got up early this morning planning  to run with the different input scalings and  
the cosine schedule that Jeremy was showing  and some of the other schedulers we look at,   I thought it might be nice for the lesson  to have a little plot of like what is the  
fit with these different solvents and input  scalings, but it was going to be a lot of work.   I was like, I'm not looking forward to doing  the groundwork. And then Tanishq sent me this  
paper which AK [@_akhaliq] had just tweeted  out because he reviews everything that comes   up on arXiv every day: “On the Importance  of Noise Scheduling for Diffusion Models”.
And this is by a researcher at the Google Brain  team who's also done a really cool recent paper  
on something called a Recurrent Interface  Network outside of the scope of this lesson,   but also worth checking out. Yeah,  so this paper they're hoping to study  
this noise scheduling and the strategies that  you take for that and they want to show that   number 1) noise scheduling is crucial for  performance and the optimal one depends on  
the task. When increasing the image size,  the noise scheduling that you want changes  
and scaling the input data by some factor  is a good strategy for working with this.
JEREMY: And that's the bit  we've been talking about, right? JOHNO: Yeah, that's what we've been doing  where we said, oh, do we scale from minus 0.5  
to 0.5 or minus 1 to 1 or do we normalize?  And so they demonstrate the effectiveness  
by training a really good high resolution  model on ImageNet. So class condition model.
JEREMY: It looks great. JOHNO: Yeah, amazing samples.  That I’ll show them later.   So I really liked this paper. It's very short  and concise and it just gets all the information  
across. And so they introduced us here. We have  this noising process our noisify function where  
we have square root of something times X plus  square root of one minus that something times  
the noise. And here they use gamma, gamma  of t, which is often used for the continuous   time case. So instead of the alpha bar and the  beta bar schedule for a thousand time saves,  
there'll be some function gamma of t that  tells you what your alpha bar should be. JEREMY: Okay. So that's how our function is  actually called abar, but it's the same thing.
JOHNO: Yeah. Same thing. Takes in a time set from  0 to 1 and then that's used to noise the image.
JEREMY: Interestingly, what they're showing here  actually is something that we had discovered and  
I've been complaining about that my DDIMs with  an eta of less than one weren't working, which is  
to say when I added extra noise to the image, it  wasn't working. And what they're showing here is  
like, oh yeah, duh, if you use a small image, then  adding extra noise is probably not a good idea 
. JOHNO: Yeah. And so they, they, they use a lot  
of reference in this paper to like information  being destroyed and signal to noise ratios,   and that's really helpful for thinking about it  because it's not something that's obvious, but  
at 64 by 64 pixels adjacent pixels might have  much less in common versus the same amount of  
noise added at a much higher resolution, the  noise kind of averages out and you can still   see a lot of the image. So yeah, that's one  thing they highlight is that the same noise  
level for different image sizes might have  it, it might be a harder or easier task.  
And so they investigate some strategies for  this. They look at the different noise schedule   functions. So we've seen the original version from  the DDPM paper. We've seen the cosine schedule  
and we've seen, I think we might look at, or  the next thing that Jeremy's going to show us,  
a sigmoid based schedule. And so they show the  continuous time versions of that and they plot  
how you can change various parameters to get  these different gamma functions or in our case,  
the alpha bar where we starting at all image,  no noise at t equals zero, moving to all noise,  
no image at t equals one, but the path that  you take, it's going to be different for   these different classes of functions and  parameters and the signal to noise ratio,  
that's what this, or the log signal to noise  ratio is going to change over that time as well.
And so that's one of the knobs we can tweak. We're  saying our diffusion model isn't training that   well. We think it might be related to the noise  schedule and so on. One of the things you could  
do is try different noise schedules, either  changing the parameters in one class of noise   schedule or switching from a linear to a cosine  to a sigmoid. And then the second strategy is  
kind of what we were doing in those experiments,  which is just to add some scaling factor to x0. JEREMY: Well we were accidentally using b of 0.5.
JOHNO: Exactly. And so that's a second dial  that you can tweak is to say keeping your  
noise schedule fixed, maybe just scale x0, which  is going to change the ratio of signal to noise.
JEREMY: And so that's what Figure 4 in c)  there is what we were accidentally doing. JOHNO: Yes. Yeah, exactly. And so see if  we can get to, oh yeah, so that again,  
changes the signal to noise for different scalings  you get. So that's fine. So they have a compound,  
they have a strategy that combines some of those  things. And this is the important part. They do   their experiments. And so they have a nice  table of investigating different schedules,  
cosine schedules and sigmoid schedules, and in  bold are the best results. And you can see for   64 by 64 images versus 128 versus 256, the best  schedule is not necessarily always the same.
And so that's like important finding number one,  depending on what your data looks like using a  
different noise schedule might be optimal. There's  no one true best schedule. There's no one value of  
beta min and beta max that's just magically  the best. Likewise for this input scaling  
at different sizes with whatever schedules they  tested and different values were kind of optimal.  
And so, yeah, it's just a really great  illustration, I guess, that this is another  
design choice that's implicit or explicitly part  of your diffusion model training and sampling is  
how are you dealing with this noise schedule?  What schedule are you following? What scaling   are you doing with your inputs? And by using  this thinking and doing these experiments,  
and they come up with a kind of rule of thumb  for how to scale the image based on image size,   they show that they can, as they increase the  resolution, they can still maintain really good  
performance. Where previously it was quite  hard to train a really large resolution  
pixel space model and they're able to  do that. They get some advantage from   their fancy Recurrent Interface Network, but  still it's kind of cool that they can say,  
look, we get state of the art high  quality and 512 by 512 or 1024 by 1024  
samples on class conditioned ImageNet and  using this approach to really like consider how  
well do you train, how many steps do we need to  take? One of the other things in this table is   that they compare it to previous approaches. Oh,  we used, you know, a third of the training steps  
and for the same other settings  and we get better performance. And just because we've chosen  that input scaling better.  
And yeah, so that's the paper. Really  nice, great work to the team. And that was-
JEREMY: I'd love to, you got up in the morning and  thought, oh, it's going to be a hassle training  
all these different models I need to train  for different input scalings and different  
sampling approaches. I might just look at Twitter   first. And then you looked at Twitter  and there was a paper saying like,  
Hey!, we just did a bunch of experiments for  different noise schedules and input scaling. JOHNO: Yeah.
JEREMY: Does your life always work that  way, Johno? That seems quite blessed. JOHNO: Yeah. It's very lucky like that. Yeah. If  you wait long enough, someone else will do it.
TANISHQ: That's why it's always that the time  when AK [@_akhaliq] starts posting on Twitter,   it's like my favorite hour of the day.  It's just for all the papers to be posted.
JEREMY: Oh, well thank you for that.  So let me switch to notebook 23  
Karras (Elucidating the Design Space of Diffusion - Based Generative Models)
because this notebook is actually largely an  implementation of some ideas from this paper that  
everybody tends to just call it Karras,  unfair ‘cause there's other people.  
But I will do it anyway, Karras paper. And the  reason we're going to look at this is because  
in this paper, the authors actually take a  much more explicit look at the question of  
input scaling. Their approach was not apparently   to accidentally put a bug in their code and  then take it out, find it worked worse and  
then just put it back in again. Their approach  was actually to think, how should things be?  
So that's an interesting approach to doing things  and I guess it works for them. So that's fine.
TANISHQ: I think our approach is more exciting. JEREMY: Yeah, exactly. Our approach is much more  fun because you never quite know what's going to  
happen. And so, yeah, in their approach,  they actually tried to say like, okay,  
given all the things that are coming into our  model, how can we have them all nicely balanced?  
So we will skip back and forth  between the notebook and the paper.  
So the start of this is all the same,  except now we are actually going to do it   minus 1 to 1 because we're not going to rely  on accidental bugs anymore, but instead we're  
going to rely on the Kerras papers, carefully  designed scaling. I say that except that I  
put a bug in the notebook as well. One of  the things that's in the Kerras paper is  
what is the standard deviation of the actual  data which I calculated for a batch. However,  
this used to say minus 0.5. I used  to do the minus 0.5 to 0.5 thing. And so this is actually the standard deviation  of the data before when it was still minus 0.5.  
So this is actually half the real standard  deviation. For reasons I don't yet understand,  
this is giving me better scaled results.  So this actually should be 0.66.  
So there's still a bug here and the bug still  seems to work better. So we've still got some   mysteries involved. So we're going to leave this.  So it's actually not 0.33, it's actually 0.66.  
Okay, so the basic idea of this  paper, actually I'll come back.  
Well let me have a little think. Yeah, okay. Now  we're going to start here. So the basic idea of  
this paper is to say, you know what, sometimes  maybe predicting the noise is a bad idea.  
So like you can either try and predict the noise  or you can try and predict the clean image and  
each of those can be a better  idea in different situations. If you're given something  which is nearly pure noise,  
you know, the model's given something which  is nearly pure noise and is then asked to   predict the noise, that's basically a waste of time because the whole thing's noise.  
If you do the opposite, which is you try to get  it predict the clean image, well then if you give  
it a clean image that's nearly clean and try to  predict the clean image, that's nearly a waste of   time as well. So you want something which is like,  regardless of how noisy the image is, you want it  
to be kind of like an equally difficult problem  to solve. And so what Karras do is they basically  
use this new thing called Cskip, which is a  number which is basically saying like, you  
know what we should do for the training target,  is not just predict the noise all the time,  
not just predict the clean image all the time,  but predict kind of lerp version of one or the  
other depending on how noisy it is. So here y is  the plain image and n is the noise. So y plus n  
is the noised image. And so if Cskip was zero,  then we would be predicting the clean image.  
And if Cskip was one, we would be predicting  y minus y, we would be predicting the noise.  
And so you can decide by picking a different   Cskip whether you're predicting  the clean image or the noise.
And so, as you can see from the way they've  written it, they make this a function. They   make it a function of sigma. Now this is  where we've got to a point now where we've  
kind of got a fairly much simpler notation.  There's no more alpha bars, no more alphas,  
no more betas, no more beta bars. There's just  a single thing called sigma. Unfortunately,   sigma is the same thing as alpha bar used to be.  Right. So we've simplified it, but we've also made  
things more confusing by using existing symbol for  something totally different. So this is alpha bar.
Okay. So there's going to be a function that says,  depending on how much noise there is, we'll either  
predict the noise or we'll predict the clean  image or we'll predict something between the two.  
So in the paper, they showed this chart where  they basically said like, okay, let's look at  
the loss to see how good are we with a trained  model at predicting when sigma is really low. So  
when there's very small alpha bar or when sigma  is in the middle or when sigma is really high.
And they basically said, you know what, when  it's nearly all noise or nearly no noise,  
you know, we're basically not  able to do anything at all.   You know, we're basically good at doing  things when there's a medium amount of noise.  
So when deciding, okay, what, what segments  are we going to send to this thing? The first  
thing we need to do is to, is to figure out  some sigmas. And they said, okay, well let's  
pick a distribution of sigmas that matches this  red curve here, right, as you can see. And so this  
is a normally distributed curve where this is on a  log scale. So this is actually a log normal curve.  
So to get the sigmas that they're going to use,  they picked a normally distributed random number  
and then they exp'd it. And this is  called a log normal distribution.  
And so they used a mean of minus 1.2 and a  standard deviation of 1.2. So that means that  
about one third of the time they're going to be  getting a number that's bigger than zero here.  
And e to the zero is one. So about one third  of the time they're going to be picking  
sigmas that are bigger than one.  And so here's a histogram I drew  
of the sigmas that we're going to  be using. And so it's nearly always,  
you know, less than five. But sometimes it's way  out here. And so it's quite hard to read these  
histograms. So this really nice library called  seaborn, which is built on top of Matplotlib,  
has some more sophisticated and often nicer  looking plots. And one of them they have is called  
a kdeplot, which is a kernel density plot. It's  a histogram, but it's smooth. And so I clipped it  
at 10 so that you could see it better. So you can  basically see that the vast majority of the time   it's going to be somewhere, you know, about 0.4  or 0.5. But sometimes it's going to be really big.
So our noisify is going to pick a sigma  using that log normal distribution. And  
then we're going to get the noise as  usual. But now we're going to calculate  
c_skip, right? Because we're going to do that  thing we just saw. We're going to find something  
between the plain image and the noised input.  So what do we use for c_skip? We calculate it  
here. And so what we do is we say, what's the  total amount of variance at some level of sigma?  
Well, it's going to be sigma squared. That's the  definition of the variance of the noise. But we  
also have the sigma of the data itself,  right? So if we add those two together,  
we'll get the total variance. And  so what the Karras paper said to do  
is to do the variance of the data divided by  the total variance and use that for c_skip. So  
that means that if your total variance is really  big, so in other words, it's got a lot of noise,  
then c_skip is going to be really  small. So if you've got a lot of noise,  
then this bit here will be really small.  So that means if there's a lot of noise,  
try to predict the original image, right? That  makes sense because predicting the noise would   be too easy. If there's hardly any noise, then  this will be, total variance will be really small,  
right? So c_skipwill be really big. And  so if there's hardly any noise, then  
try to predict the noise. And so  that's basically what this c_skip does.
So it's a kind of slightly weird idea is that our  target, the thing we're trying to do actually is  
not the input image, sorry, the original image.  It's not the noise, but it's somewhere between  
the two. And I've found the easiest way to  understand that is to draw a picture of it.  
So here is some examples of noised  input, right? With various amounts  
Picture of target images
of… with various sigmas. Remember sigma  is alpha bar, right? So here's an example  
with very little noise, 0.06.  And so in this case, the target  
is predict the noise, right? So that's the  hard thing to do, is predict the noise.  
Or else here's an example, 4.53, which is nearly  all noise. So for nearly all noise, the target is  
predict the image, right? And then for something  which is a little bit between the two, like here,  
0.64, the target is predict some  of the noise and some of the image.
So that's the idea of Karras. And so  
what this does is it's making the  problem to be solved by the UNet  
equally difficult, regardless of what sigma is.  It doesn't solve our input scaling problem. It  
solves our kind of difficulty scaling problem.  To solve the input scaling problem, they do it.
TANISHQ: I just want to make one quick note. And  so this idea of interpolating between the noise  
and the image is similar to what's called the  V-Objective as well. So there's also a similar  
kind of, it's quite similar to what Karras  et al. has. But that's also now been used  
in a lot of different models. For example,  Stable Diffusion 2.0 was trained with this  
sort of V-Objective. So people are using this  sort of methodology and getting good results.  
So it's an actual practical thing that people are  doing. So yeah, just want to make a note of that. JEREMY: Yeah. As is the case of basically  all papers created by NVIDIA researchers,  
of which this is one, it flies under  the radar and everybody ignores it.  
The V-Objective paper came from the senior  author, was Tim Salimans, which is Google, right?  
So anything from Google and OpenAI, everybody  listens to. So yeah, although Karras, I think   has done the more complete version of this.  And in fact, the V-Objective was almost like  
mentioned in passing in the  distillation paper. But yeah,   that's the one that everybody has ended up  looking at. But I think this is the more complete…
TANISHQ: I think what happened with  the V-Objective is not many people   paid attention to it. I think folks like Kat  and Robin, these sorts of folks are actually  
paying attention to that V-Objective  in that Google Brain paper. But then   also this paper did a much more principled  analysis of this sort of thing. So yeah,  
I think it's very interesting how, yeah. Sometimes  even these sort of side notes in papers that maybe  
people don't pay much attention to,  they can actually be quite important. JEREMY: Yeah. Yeah. So okay, so the noise input  as usual is the input image plus the noise times  
Scaling problem - (scalings)
the sigma. But then, and then as we discussed, we  decide how to kind of decide what our target is.  
But then we actually take that noise input  and we scale it up or down by this number.  
And the target, we also scale up or down by this  number. And those are both calculated in this  
thing as well. So here's c_out   and here's c_in. Now I just wanted to show one  example of where these numbers come from because  
for a while they all seem pretty mysterious to  me and I felt like I'd never be smart enough   to understand them, particularly because they  were explained in the mathematical appendix  
of this paper, which are always the bits I  don't understand, until I actually try to,   and then it tends to turn out they're not so bad  after all, which was certainly the case here.
TANISHQ: I think it was up, it was B something  I think. So the B6 I think, is the other one.
JEREMY: Oh, yeah. So in appendix B6,  which does look pretty terrifying,  
but if you actually look at, for example,  what we were just talking about, c_in,   it's like, how do they calculate? So c_in is  this. Now this is the variance of the noise,  
this is the variance of the data, add them  together to get the total variance, square roots,   the total standard deviation. So it's just the  inverse of the total standard deviation, which is  
what we have here. Where does that come  from? Well they just said, you know what?  
The inputs for a model should have unit variance.  Now we know that. We've done that to dare  
in this course. So they just said, all right,  so well the inputs to the model is the clean  
data plus the noise times some number we're  going to calculate, and we want that to be one.  
Okay, so the variance of the clean images plus the  noise is equal to the variance of the clean images  
plus the variance of the noise.  Okay, so if we want that to be,  
if we want variance to be one, then divide  both sides by this and take the square root,  
and that tells us that our multiplier has to be  one over this. That's it. So it's like literally,  
you know, classical math. The only bit you have  to know is that the variance of two things added  
together is the variance of the two things added  together, which is not rocket science either.
JOHNO: And in this context, like why we want to  do this?, when we looked at those sigma's that   you're plotting, like the distribution, you've  got some that are fairly low, but you've also  
got some where the standard deviation sigma is  like 40, right? So the variance is super high. JEREMY: Yes. JOHNO: And so we don't want to feed something with  standard deviation 40 into our model. You would  
like it to be closer to unit variance. So we're  thinking, okay, well, if you divide by roughly 40,  
that would scale it down. But then we've also got  some extra variance from our data. It's just like  
40 plus variance of the data of a little bit. We   want to scale back down by  that to get unit variance.
JEREMY: Yeah. I mean, I love this paper  because it's basically just doing what we spent  
weeks doing. I feel like everything that  we've done that's improved every model has  
always been one thing, which is, can we get  mean zero, variance one inputs to our model  
and for all of our activations? And then the  only other thing is include enough compute by  
adding enough layers and enough activations.  Those two things seem to be all that matters.  
Basically, well, I guess ResNets added an extra  cool little thing to that, which is to make it  
even smoother by giving this kind of like identity  path. So yeah, basically trying to make things as  
smooth as possible and as equal  everywhere as possible. So yeah,  
this is what they've done. So they did that  for the inputs and then they've also done it  
for the outputs. And for the  outputs, you know, it's basically  
the same idea, you know, and they have basically  the same kind of analysis to show that.  
And so with this, so now, yeah, we've  basically, we've got our noised input.  
We've got the, you know, kind of linear version  somewhere between X nought and the noise to input.  
We've got the scaling of the output and we've got  the scaling of the input. So now for the inputs to  
our model, we're going to have the scaled noise.  We're going to have the sigma and we're going  
to have the target, which is somewhere between  the image and the noise. And so, yeah, so I've,  
you know, never seen anybody draw a picture  of this before. So it was really cool when,   you know, being in a notebook, being able to see  like, oh, that's what they're doing, you know?  
So yeah, have a good look at this notebook  to see exactly what's going on. Cause I   think it gives you a really good intuition  around what problem it's trying to solve.
So then I actually checked the noised input has  a standard deviation of one, the means not zero.  
And of course, why would it be? We didn't  do anything, you know, the only thing Karras   cared about was having the variance one. We could  easily adjust the input and output to have a mean  
of zero as well. That's something I think we or  somebody should try. Cause I think it does seem  
to help a bit as we saw with that generalized  value stuff we did. But it's less important   than the variance. And so same with the target.  It's got the one and yeah, this is where if I  
changed this to the correct value, which is 0.66,  then actually it's slightly further away from one  
both here and here, quite a lot further away. And  maybe that's because actually the data's well,  
we know the data's not Gaussian distributed pixel  data definitely isn't Gaussian distributed. So  
this bug turned out better. Okay. So the unit's the same, the  initialization's the same. This is all the same.  
Training and predictions of modified model
Train it for a while. We can't compare the  losses, right? Because our target's different. So,  
but what we can do is we can create  a denoise that just takes the  
thing that as, per usual, the thing  we had in noisify, right? And so for  
x naught, so you're going to multiply by  c_out and then add c_skip by noised_input.   Here it is, multiply by c_out, add noised_input,  c_skip. Okay. So we can denoise. So let's grab our  
sigmas from the actual batch we had. Let's  calculate c_skip, c_out and c_in for the sigmas  
in our mini batch. Let's use the model to predict  the target given the noise to input and the sigmas  
and then denoise it. And so here's our noise to  input, which we've already seen, and here's our  
predictions. And these are  absolutely remarkable in my opinion.  
Yeah. Like this one here, I can barely see it.  You know, it's really found. Look at the shirt.  
There's the shirt here. It's actually really  finding the little thing on the front and let me   show you. Here's what it should look like. And in  cases where the sigma is pretty high, like here,  
you can see it's really like saying like, I  don't know, maybe it's shoes, but it could be   something else. Is it shoes? Yeah, it wasn't  shoes, but at least it's kind of got the,  
you know, the bulk of the pixels in  the right spot. Yeah. Something like   this one is 4.5. Has no idea what it is. It's  like, oh, maybe it's shoes. Maybe it's pants.  
You know, it turns out it is shoes. Yeah. So  I think that's fascinating how well it can do.
And then the other thing I did, which I  thought was fun was I just created, so I just,  
you did a sigma of 80, which is actually what they  do when they're doing sampling from pure noise.   That's what they consider the pure noise level.  So I just created some pure noise and denoised it  
just for one step. And so here's what happens  when you denoise it for one step. And you can see  
it's kind of overlaid all the possibilities.  It's like, I can see a pair of shoes here,   a pair of pants here at top here. And sometimes  it's kind of like more confident that the noise  
is actually a pair of pants. And sometimes  it's more confident that it's actually shoes.   But you can really get a sense of how like from  pure noise, it starts to make a call about like  
what this noise is actually covering up. And  this is also the bit which I feel is like,  
I'm the least convinced about when it comes to  diffusion models. This first step of going from  
like pure noise to something and like trying to  have a good mix of all the possible somethings.  
I don't know, it feels a bit handwavy to me.  It clearly works quite well, but I'm not sure   if it's like we're getting the full range of  possibilities. And I feel like some of the  
papers we're starting to see is starting to say  like, you know what, maybe this is not quite the   right approach. Then maybe later in the course,  we'll look at some of the ones that look at  
what we call VQ models and tokenized stuff.  Anyway, I thought this was pretty interesting  
to see these pictures, which I don't think,  yeah, I've never seen any pictures like this   before. So I think this is a fun result from  doing all this stuff in notebooks step by step.
Okay, so sampling. So one of the nice things  with this is the sampling becomes much,  
Sampling
much, much simpler. And so, and I should mention  a lot of the code that I'm using, particularly  
in the sampling section is heavily inspired by,  and some of it's actually copied and pasted from  
Kat's k-diffusion repo, which is, I think  I mentioned before, some of the nicest  
generative modeling code or maybe the nicest  generative modeling code I've ever seen. It's   really great. So before we talk about the actual  sampling, the first thing we need to talk about is  
what sigma do we use at each reverse time step.  And in the past, we've always, well, nearly   always done something, which I think has always  felt is sketchy as all hell, which is we've just  
linearly gone down the sigmas or the alpha bars  or the t’s. So here, when we're sampling in the  
previous notebook, we used linspace. So I always  felt like that was questionable. And I felt like  
at the start, you probably like it was just  noise anyway. So who cared? Who cares? So I,  
in DDPM_v3, I experimented with something  that I thought intuitively made more sense.  
I don't know if you remember this one, but I  actually said, oh, let's, for the first hundred  
time steps, let's actually only run the model  every 10 times. And then for the next hundred,   let's run it nine times. The next one hundred,  let's run it every eight times. So basically at  
the start, be much less careful. And so Karras  actually ran a whole bunch of experiments. And  
they said, yeah, you know what, at the start of  training, you know, you can start with a high  
sigma, but then like step to a much lower sigma  in the next step and then a much lower sigma in  
the next step. And then the longer, the more you  train step by smaller and smaller steps so that  
you spend a lot more time fine tuning carefully  at the end and not very much time at the start.
Now, this has its own problems. And in  fact, a paper just came out today, which  
Sampling: Problems of composition
we probably won't talk about today, but maybe  another time, which talked about the problems   is that in these very early steps, this is the  bit where you're trying to create a composition  
that makes sense. Now for Fashion-MNIST, we  don't have much composing to do. It's just a   piece of clothing. But if you're trying to  do an astronaut riding a horse, you know,  
you've got to think about how all those pieces fit  together. And this is where that happens. And so  
I do worry that with the Karras approach is not  giving that maybe enough time. But as I've said,  
that's really the same as this step at  that, that whole piece feels a bit wrong  
to me. But aside from that, I think this makes a  lot of sense, which is that, yeah, the sampling,  
you should jump, you know, by big steps early  on and small steps later on and make sure that  
the fine details are just so. So that's what  this function does, is it creates this plot.
Now it's this schedule of reverse diffusion  sigma steps. It's a bit of a weird function  
in that it's the, the rho-th root of sigma,  where rho is seven. So the seventh root of  
sigma is basically what it's scaling on. But  the answer to why it's that is because they  
tried it and it turned out to work pretty  well. Do you guys remember where this was?
Sampling: Rationale for rho selection
TANISHQ: This is the  truncation error analysis, D1.
JEREMY: Nice memory. So this image here –so thanks  for Tanishq reminding me where this is– shows FID  
as a function of rho. So it's basically what, the  what root are we taking. And they basically said,  
like, if you take the fifth root up,  it seems to work well, basically.  
So yeah, so that's a perfectly good way to do  things is just to try things and see what works.  
And you'll notice they tried things,  just like we love, on small datasets,   not as small as us because we're the king of small  datasets, but smallish, CIFAR-10, ImageNet-64.  
That's the way to do things. So I saw like –it  might've even been the CEO of Hugging Face the  
other day– tweets something saying only people  with huge amounts of GPUs can do research now.   And I think it totally misunderstands how research  is done, which is research is done on very small  
datasets. That's the actual research. And then  when you're all done, you scale it up at the end.  
I think we're kind of pushing the  envelope in terms of like, yeah, how   much can you do? And yeah, we've like  re-covered this kind of main substantive path of  
diffusion models history step-by-step showing  every improvement and seeing clear improvements  
across all the papers using nothing but  Fashion-MNIST running on a single GPU in like  
15 minutes of training or something per model.  So yeah, definitely don't need lots of models.
Anyway. Okay. So this is the sigma we're going  to jump to. So the denoising is going to involve  
Sampling: Denosing
calculating the c_skip, c_out and c_in and calling  our model with the c_in scaled data and the sigma  
and then scaling it with c_out and then doing the  c_skip. Okay. So that's just undoing the noisify.  
So check this out. There's all that's required to  do one step of denoising for the simplest kind of  
scheduler, which is sorry, the simplest  kind of sampler, which is called Euler.   So we basically say, okay,  what's the sigma at time step i,  
what's the sigma two at time step i. And  now when I'm talking about time step,   I'm really talking about like the step from  this function, right? So this is, this is –
JOHNO: Sampling step. JEREMY: Sampling step, yeah. Okay.  So then denoise –using the function–  
and then we say, okay, well just  send back whatever you were given,  
plus move a little bit in the direction of the  denoised image. So the direction is x minus  
denoised. So that's the noise, that's the gradient  as we discussed right back in the first lesson of  
this part. So we'll take the noise. If we divide  it by sigma, we get a slope. That's how much  
noise is there per sigma. And then the amount  that we're stepping is sigma two minus sigma  
one. So take that slope and multiply it by the  change, right? So that's the distance to travel  
towards the noise, that this fraction, you  know, or you could also think of it this  
way. And I know this is a very obvious  algebraic change, but if we move this   over here, you could also think of this as  being, oh, of the total amount of noise,  
the change in sigma we're doing, what  percentage is that? Okay, well that's the amount  
we should step. Right? So there's two  ways of thinking about the same thing.   So again, this is just, you know,  high school math. Well I mean,  
actually my seven year old daughter has done all  these things. It's plus minus dividing times.
So we're going to need to do this once per  sampling step. So here's a thing called sample,  
which does that. That's going to  go through each sampling step,  
call our sampler, which initially we're going to  do sample Euler, right? With that information,  
add it to our list of results and do it again.  So that's it. That's all the sampling is.  
And of course we need to grab  our list of sigmas to start with.   So I think that's pretty cool. And at the very  start we need to create our pure noise image.  
And so the amount of noise we  start with is got a sigma of 80.  
Okay, so if we call sample using sample Euler,  and we get back some very nice looking images  
and, believe it or not, our FID is 1.98. So this  extremely simple sampler, three lines of code plus  
a loop has given us a FID of 1.98, which is  clearly substantially better than our coastline.
Now we can improve it from there. So one  potential improvement is to, you might've  
noticed we added no new noise at all, right? This  is a deterministic scheduler, right? There's no  
rand anywhere here. So we can do something called  an Ancestral Euler Sampler, which does add rand,  
right? So we basically do the denoising in the  usual way, but then we also add some rand. And  
so what we do need to make sure is given that  we're adding a certain amount of randomness,   we need to remove that amount of randomness  from the step that we take. So I won't go  
into the details, but basically there's our way of  calculating how much new randomness and how much  
just going back in the existing direction do we  do. And so there's the amount in the existing  
direction and there's the amount in the new  random direction. And you can just pass in eta,  
which is just going to, when we pass it into here,  is going to scale that. So if we scale it by half,  
so basically half of it is new noise and  half of it is going in the direction that   we thought we should go, that makes it better  still. Again with a hundred steps and just make  
sure I'm comparing to the same. Yep. A hundred  steps. Okay. So it's fair. Like with like.  
Okay. So that's adding a bit of extra noise. Now then the… something that I think we might've  mentioned back in the first lesson of this part  
is something called Heun's method.   And Heun's method does something which we can  pictorially see here to decide where to go,  
Sampling: Heun’s method fid: 0.972
which is basically we say, okay, where  are we right now? What's the, you know,  
at our current point, what's the direction? So we  take the tangent line, the slope, right? That's  
basically all it does is it takes a slope. So  it's not, here's a slope, you know? Okay. And so  
if we take that slope and that  would take us to a new spot  
and then at that new spot, we can then  calculate a slope at the new spot as well.  
And at the new spot, the slope is something else.  So that's it here, right? And then you say like,  
okay, well, let's go halfway between the two  and let's actually follow that line. And so  
basically it's saying like, okay, each of  these slopes is going to be inaccurate.  
But what we could do is calculate the slope of  where we are, the slope of where we're going   and then go halfway between the two. I actually  find it easier to look at in code personally.  
I just kind of delete a whole bunch of stuff  that's totally irrelevant to this conversation.
So take a look at this compared to Euler.  
So here's our Euler, right? So we're going  to do the same first line exactly the same,   right? Then the denoising is exactly the same,  right? And then this step here is exactly the  
same. I've actually just done it in multiple  steps for no particular reason. And then say,  
okay, well, if this is the last step, then we're  done. So actually the last step is Euler. But then  
what we do is we then say, well, that's okay  for an Euler step, this is where we'd go.  
Well, what does that look like if we denoise  it? So this calls the model the second time,  
right? And where would that take us if we took  an Euler step there? And so here, if we took  
an Euler step there, what's the slope? And so what  we then do is we say, oh, okay, well, it's just,  
just like in the picture, let's take the average.  Okay, so let's take the average and then use that,  
the step. So that's all the Heun sampler  does is it just takes the average of the  
slope where we're at and the slope where  the Euler method would have taken us. And so if we now, so notice that it called the  model twice for a single step. So to be fair,  
since we've been taking a hundred steps with  Euler, we should take 50 steps with Heun,   right? Because it's going to call the model twice.  And still that is now, whoa, we beat 1, which  
is pretty amazing. And so we could keep going,  check this out. We can even go down to 20. This  
is actually doing 40 model evaluations and this is  better than our best Euler, which is pretty crazy.
Now something which you might've noticed is kind  of weird about this or kind of silly about this  
is we're cat–, we're calling the model twice just  in order to average them, but we already have two  
Sampling: LMS sampler
model results like without calling it twice. We  cause we could have just looked at the previous   time step. And so something called the LMS  sampler does that instead. And so the LMS sampler,  
if I call it with 20, it actually literally  does 20 evaluations and actually it  
beats Euler with a hundred evaluations. And so  LMS, I won't go into the details too much. It  
didn't actually fit into my little sampling very  well. So basically largely copied and pasted the   Kat's code. But the key thing it does is look  at, it gets the current sig –sigma. It does the  
denoising, it calculates the slope and it stores  the slope in a list, right? And then it grabs  
the first one from the list. So it's kind of  keeping a list of up to this case four at a time.  
Kerras Summary
And so then uses up to the last four to basically,  yes, kind of the curvature of this and take the  
next step. So that's pretty smart. And yeah, so  I think if you wanted to do super fast sampling,  
it seems like a pretty good way to do it.  And I think Johno, you were telling me that,  
or maybe it's Pedro was saying that  currently people have started to move away,   that this was very popular, but people  started to move towards a new sampler,  
which is a bit similar called the  DPM++ sampler, something like that.
TANISHQ: Yeah. Yeah. Yeah. JOHNO: Yeah. JEREMY: But I think it's  the same idea. So it kind of  
keeps a list of recent results and use  that. I'll have to check it more closely.
JOHNO: The similar idea is like, if it's  done more than one step, then it's using  
some history to the next thing. JEREMY: Yeah. [unintelligible] in Heun  doesn't make a huge amount of sense, I guess,  
from that perspective. I mean, still works very  well. This makes more sense. So then, we can  
compare if we use an actual mini-match of data,  we get about 0.5. So yeah, I feel like this is  
quite a stunning result to get  
very close to real data, at least in terms of  FID. You know, really with 40 model evaluations.  
And the entire, nearly the entire thing here is  by making sure we've got unit variance inputs,  
unit variance outputs, and kind of equally  difficult problems to solve in our loss function.
JOHNO: Yeah. Plus having that different schedule  for sampling, that's completely unrelated to the  
training schedule. So one of the big things  with Karras et al's paper was they also could  
apply this to like, oh, existing diffusion  models that have been trained by other papers,   we can use our sampler and in fewer steps  get better results without any of the other  
changes. And yeah, I mean, they do a  little bit of rearranging equations  
to get the other papers versions into  their c_skip, c_in, c_out framework.  
But then, yeah, it's really nice that these  ideas can be applied to. So for example, I  
think stable diffusion, especially version one was  trained DDPM style training, epsilon objective,  
whatever. But you can now get these different  samplers and different something schedules and  
things like that and use that to sample it and do  it in 15, 20 steps and get pretty nice samples.
JEREMY: Yeah. You know, and another  nice thing about this paper is  
Comparison of different approaches
they, you know, in fact, it's the name of  the paper, “Elucidating the Design Space of  
Diffusion-Based …” models. You know, they looked  at various different papers and approaches and  
trying to set like, oh, you know what? These  are all doing the same thing when we kind of  
parameterize things in this way. And if you fill  in these parameters, you get this paper and these   parameters, you get that paper, you know? And  then, so, we found a better set of parameters,  
which… it was very nice to code because, you know,  it really actually ended up simplifying things  
a whole lot. And so if you look through the  notebook carefully, which I hope everybody will,  
you'll see, you know, that the code  is really there and simple compared to  
previous, all the previous ones, in  my opinion. Like I feel like every   notebook we've done from DDPM onwards,  the code's got easier to understand  
and the results… TANISHQ: And just to, again, clarify, like,  how this connects with some of the previous   papers that we've looked at. So like, for example,  with the DDIM, the deterministic, that's again,  
the deterministic approach, that's similar to the  Euler method sampler that we were just looking at,  
which was completely deterministic. And then  some of something like the Euler Ancestral   that we were looking at is similar to  the standard DDPM approach with the,  
that was kind of a more stochastic approach.  So, again, there's just all these sorts of   connections that then are kind of nice to  see, again, the sorts of connections between  
the different papers and how they change it, how  they can be expressed in this common framework.
JEREMY: Yeah. Thanks, Tanishq. So we definitely  now are at the point where we can show you the  
Next lessons
UNet next time. And so I think we're, unless any  of us come up with interesting new insights on  
the unconditional diffusion sampling, training  and sampling process, we might be putting that  
aside for a while. And instead we're going  to be looking at creating a good quality UNet  
from scratch. And we're going to look at a  different data set to do that. This was starting  
to scale things up a bit as Johno mentioned in  the last lesson. So we're going to be using a 64  
by 64 pixel ImageNet subset, called Tiny ImageNet.  So we'll start looking at some 3-channel images.  
So I'm sure we're all sick of looking at black  and white shoes. So now we get to look at shift  
dwellings and trolley buses and koala bears and  yeah, 200 different things. So that'll be nice.  
Yeah. All right. Well, thank you, Johno. Thank  you, Tanishq. That was fun as always. And  
next time will be Lesson 22. Bye. JOHNO: This was Lesson 22.
JEREMY: Oh, no way. Okay. See you.

---

# 23

Hi everybody, today we are covering lesson 23  and we're here with Johno and Tanishq. How are  
you guys both doing? Doing well? Yes, I'm  doing well. I'm saying for another lecture.  
Yeah, likewise. Great. I, shamefully,  have to start with admitting to a bug,  
which actually is rather... Well, I don't know.  It kind of messed up things in a sense, but I  
kind of, I think it's really interesting actually  what happened. The bug, it was in Notebook 23,  
the Karras Notebook, and it's about the measuring  the FID. So, to recall, FID measures how similar  
a bunch of samples are from a model to  a bunch of samples of real images, and  
that similarity is defined in this kind of like  some kind of distance between the distributions  
of the features in a classifier or some kind of  model. So, that means that to get FID, we have to  
load a model, and we have to pass it some  data loaders so that it can calculate what  
the samples look like from real images. Now, the  problem is that the data loaders I was passing  
actually had images that the pixels were  between negative 0.5 and positive 0.5. But  
you might recall this model that I trained has  pixels between negative 1 and 1. So, what this  
image, eval class, would have seen, and  specifically this sea model, which we are putting,   which we are getting the features from, is it  would have seen a whole bunch of unusually low  
contrast images. So, they wouldn't really have  looked like many things in the data set, because  
in fact in the data set, I think, particularly  for fashion emnest things are pretty consistently  
normalized in terms of going all the way from  0 to 1 or negative 1 to 1, or I guess 0 to 255  
in the original. And so, as a result, I think  what would have happened is that the features  
that came out of this would have been kind  of weird, and they might not have necessarily   consistently said, oh, these are t-shirt features  and these are two features, but they would have  
said, oh, this is a weird locust low contrast  image feature. And so, then the shame continues  
in that I added another bug on top of this bug,  which is when I then did the sampling, I didn't,  
I didn't multiply by 2, and the data that  I trained it on was actually the same data  
loaders, well, the specifically the same  transform, the same noise if I transform.  
Well, where did it come from? It's the same,  yeah, the same transform I, not noise if I,  
the same transform I, which, yeah, previously  was point from point negative 0.5 to 0.5. So,   I trained the model using this restricted input  space as well. And therefore, it was spitting  
out things that were between negative point 5 and  point 5. And so, the FID then said, wow, these are  
so similar, the samples are consistently spitting  out features of low contrast things and all of the  
real samples are low contrast things. So, those  are really similar. And that's how we got really   low numbers. So, those lung numbers are wrong. So,  I was a bit surprised, I guess, that the carous  
model was doing so much better, and it certainly  it made me a big believer in the carous model,  
but actually it's not doing so much better. So,  once we fix that, the FIDs are actually around  
5, 6, 5, and the reals are 2.5. So, to compare,  
we were getting some pretty good results in  cosine. So, cosine, yeah, we were getting 3,  
4, depending on how many steps we were doing DDIM.  
So, the result of this is that this somewhat  odd situation where the cosine model, where we  
scaled it accidentally to be negative point 5  to point 5. And then, post sampling, multiply  
by 2. So, we're not cheating, like the carous  one used to be is working better than carous,  
which, yeah, is a surprise to me, because  I was thinking carous was kind of like  
in theory, optimally scaling things. But I  guess the truth is, it was scaling things to  
unit variance, but there's nothing particularly  to say that's optimally scaling things and so  
empirically we've found kind of accidentally  a better way to scale things. And also, our  
dependent variable is different. Our  dependent variable is not that carous,   you know, c, mix combination, but our dependent  variable is just the noise, the 0, 1 noise,  
you know, the noise before it's multiplied by  alpha bar. Okay, so that's, that's the bug.  
Anyway, I promised last time we would stop  looking at fashion, MNIST through a while. So,   let's move on to Tiny Imagenet. So, and the reason  we're going to do this is because we want to,  
I want to show an example of, we're going to try  and create Unets today. And I wanted to show it  
example of a, of a nice Unet we can create the  combines a lot of the ideas we've been looking at.  
It's going to be a super resolution Unet  and doing super resolution on fashion,   MNIST isn't going to be very interesting because  the maximum training size we have is 28 by 28. So,  
so I thought we'd go a little bit bigger than  that to Tiny Imagenet, which is 64 by 64.  
I found it quite difficult actually to find Tiny  Imagenet data, but eventually I discovered that  
it's still on the Stanford servers were originally  created. It's just not linked to anywhere.  
So, we'll try to keep if this disappears, we  will, we will keep our forum and website up   to date with other places to find it. Anyway,  so for now, we can grab the URL from there and  
unpack it. So, SHUTIL is a very handy little  library inside the Python standard library and  
one of the things that has is a very handy  unpack archive, switching handles it files   and it's going to put it in our data directory.  So, I, yeah, there's a few different ways we  
could process this and I thought we might  experiment with some things, but I thought,   yeah, it wouldn't be a bad idea to try doing  things the reasonably kind of manual way  
just to see what that looks like. And often  this is the easiest way to do things because  
you know, that's a very well defined set of steps,  right? So, step one is to create a data set. So,  
a data set is just literally something that has  a length and that you can index into it. So,  
it has to have these two things to find.  You don't have to inherit from anything,   you just have to define these two things. Broadly  speaking in Python, you generally don't have to  
inherit from things. You just have to provide  the methods that are expected. So, our data set  
is in a directory called Tiny Imagenet 200  and then there's a train directory and a  
valve directory for the trading and the validation  set and then the train directory, this is pretty   classic normal thing each category. So, this is  a category as images in a separate folder and  
specifically there are images subfolder. So, what  I wanted to do was to just grab start with grab  
all of the files in path slash train or the image  files. So, the Python standard library has a glob  
function, which searches recursively if asked  to for everything that matches this, well, this  
specification. So, this specification is path  slash start at jpeg and then this star star here,  
I don't know why we need to do it twice, it's a  bit weird, it was that you also need that to be   recursive. So, to be recursive, you both have  to say recursive tree here and also put star  
star before the slash here. So, it's going to  give us a list of all files inside path train.  
And so, then if we index into that training data  set with zero, that will call get item passing  
an eye of zero. And so we will then return  a tuple. One is the thing in self.files i,  
which is this file and then the label for it and  the label is that. So, it's the parents parents  
name parents parents name. And so that's the  name. Okay, so there's a data set that returns two  
strings when you index into it, a couple of two  strings. The first is the name of the image file,  
so the path of the image file and the second is  the name of the category it's in. These weird  
names are called word net categories, there's like  codes that indicate concepts basically in English.  
So, one of the reasons I actually used this  particular data set is because it's going  
to force us to do some more data processing,  which I think is good practice. That's because  
weirdly in the validation set, although it's  in Tiny Imagenet 200 slash val, which is the  
not weird part, the weird part is that they  are not then in sub directories organized by  
label instead there is a separate  val annotations. And file,  
which looks like this, so it says to each  file name. What category is it is also got  
the like the bounding box of whereabouts that  is, but we're not going to be using that today.  
I decided to create a dictionary that will tell  us for each file, what categories it in. So that  
means that I want to create a this case here, I'm  doing something exactly like a list comprehension,  
but because it's not in square brackets, it's  a generator comprehension, so it'll generate.   You've kind of stream out the results and  we're going to go through each line in.  
This file. And we're going to split on tab.  So that's going to give us this and this  
and this and then we're going to grab the  first two. And if you basically pass a, a  
list of lists or list of tuples or whatever  to dict it will create a dictionary  
using these pairs as key values. So if we have a  look. There it is. That's quite a nice neat way  
to do it. And if you're not sure you can  just click type, deck type open brackets  
and then hit shift tab a couple of times and  it'll show you the various options. And you  
can see here I'm doing dict iterable because my  generator is it is it is iterable and it says,   oh, that's exactly as if you created a dictionary  and then gone for KV in iterable decay equals V.  
So there's a nice little trick. Now  
we need a data set that works just like tiny data  set, but the get items are going to label things  
differently. So I just inherited from tiny data  set. So that means we don't need to do in it or  
then again and then get item again. It's going  to turn the file. This time the label will not  
be the parent parent name, but we will look up in  the annotations dictionary. The name of the file.  
And so that works. We can check the length  works. So then a fairly generally useful  
thing that I thought will then create  is something that lets us transform  
any data set. So here's a class that you can  pass it to data set. And you can pass it a  
transformation for the X to the independent  variable and you can pass it a transformation   for the way. And both of them default to know up  that is no operation. So it just doesn't change  
it at all. So a transform data set. The length  of it is just the length of the original data  
set. So we will get item. It'll grab the tuple  from the data set we passed in and it will return  
that tuple but with transform X and transform  Y applied to it. Does that make sense so far?  
Great. Okay. So I don't like working  with these and zero three zero things,  
but the data set luckily has  a word net IDs file in it.  
So if I just open it up. Oh, sorry, this one  actually is not quite going to help us. This  
is just a list of all of the word net IDs that  they have images for. We could have actually  
got this by simply grabbing. By listing this  directory. It would have told us all the IDs,  
but they've got they've also got just a text file  containing all of them. We can see that there are  
200 categories. Okay. And that's useful  because we're going to want to change  
N zero three zero, etc into an int. And  the way we can change it into an int  
is by simply saying, oh, we'll call call this one  zero and this one one and so forth. So the kind  
of the int to string or ID to string version  of this is literally this list. So zero will  
be there that. But the string to int version, we  do this all the time. It's basically a numerate.  
So that gives us the index and the value for  everything in the list. So those are going to  
be their keys and values, but actually we're  going to invert it to become value column key.  
And that's what's true to ID  will be so note here that we   have a dictionary comprehension. You  can tell us it's got curly brackets  
and a colon. And so here's our dictionary  comprehension. So we could have used that  
for this as well. We could have done a dictionary  comprehension instead. But yeah, so there's lots  
of ways of doing things. None of them's any better  or worse than any other. Okay, so that's the. The   word that tags whatever do we have the names for  them or is that some? Yes, the names I'm going to  
get to. Yes, shortly. There's a word that text.  So yeah. All right, I grabbed one batch of data  
and grabbed its main and standard deviation. And  so then I've just copied and pasted them in here  
for normalizing. So my transform X is going to  be I'm going to read the image. If you read it as  
RGB that's going to force it to be three channels  because actually some of them are any one channel.   We write it by two fifty five sort of between  zero and one. And then we will normalize.  
And then for our wise, we will go through  this dread ID to get the ID and just use  
that as our tensor. So it's, you know, doing  it manually is actually pretty straightforward  
right because now we just pass those to  our to from DS our transform to data set.  
And we can check that. You  know, you can see why I.  
Is a tensor, but we can look it up to  get its value and XI is an image tensor.  
With three channels. So channel by height  by which has is normal for paytorch.  
So for showing images, it's nice to  denomalize them. So that's just denomalizing.  
And so if we show the image that we  described, it say what a joke, I guess.  
Alright, so now we can create a data loader for  our training set. So it's going to contain our  
transformed training data set and pass in  a batch size. This one has to be shuffled.  
Not sure why I put num work  as it calls zero there.   Generally, it's pretty good if  you've got at least eight cause.  
Yeah, so we can now grab an expatch and a Y batch.   And take a look at a denomalized image from  there. So there we've got a nice little kitty cat.  
So I think this is already looking  better than fashion, amnest.  
Yeah, so there's this thing. So let's dot   text that they've also provided. And  this is actually a list of the entire  
word net hierarchy. So the top of the hierarchy is  entity and one of the entity types is a physical  
entity or an abstract entity entities can be  things. And so forth. So this is how word net is.  
And so this is quite a big file actually. So if  we go through each item of that file and again  
split on tabs. Because split on tabs, that's what  backslash team means is going to give us the word  
net ID and then the name of it. So now we can go  through all of those. We call them since sets.  
And if the key is in our list of  the 200 that we want, we'll keep it.   And we don't really want like causal agent,  comma cause, comma causal agency. The first  
one generally seems to be the most normal. So  I just split on comma. And grab the first one.  
All right. So that's so we could then go  through our Y batch and just turn each of  
those numbers into strings and then look at each  of those up in our sin sets and join them up.  
And then use those as titles to see our  Egyptian cat in our cliff, the guacamole.  
So monarch butterfly. And so forth. And you can  see that this is going to be quite tricky because  
like a cliff, this is a cliff dwelling, for  instance, could be quite, you know, complicated.  
I have a feeling for this day intentionally  like a hundred of the. So I think that the  
number of degrees might have come from the normal  Imagenet. And I think they might have been picked   a hundred that are designed to be particularly  difficult or something if memory says correctly.  
All right. So then we could define a transform  batch function with the same basic idea.  
And that's just going to, yeah,  transform the X and the Y in a batch.  
Oh, yes, we're about to use that. I should move  that down a bit because we're not quite there   yet. Okay. So before that, we can create our data  loaders. We created a get deals back in an earlier  
lesson, which simply turns that into a data loader  and that into a data loader and this one gets  
shuffled and that one doesn't and so forth. Oh, I  see this is where we do our number workers. Cool.  
All right. So then. Oh, yes. So then we want  to add our data augmentation. So I noticed that  
training a Tiny Imagenet model. I mean, it's it's  a much harder thing to do than fashion, MNIST. And  
overfitting was actually a real challenge.  And I guess it's because 64 by 64 isn't that  
many pixels. So yeah, so I found I really needed  data augmentation to make much progress at all.  
Now, very common data augmentation is called  random resize crop, which is basically to pick  
like one area inside and that zoom into it and  make that your image. But for such low resolution  
images that tends to work really poorly. Because  it's going to introduce a lot of kind of blurring  
artifacts. So instead for small images, I think  it's better to add a bit of padding around them.  
And then randomly pick a 64 by 64 area from that  padded area. So it's just going to shift them  
slightly. It's not a lot of augmentation, but it's  something. And then we do our random horizontal   flips. And then we'll use that random arrays  thing that we created earlier. This is just  
something I was experimenting with. So yeah,  so now we can use that batch transform callback  
using transform batch passing in those transforms.  So with torch vision transforms, so this capital  
T is torch vision transforms. Yeah, because these  are all. And then dot modules, you can pass them  
to nn dot sequential to just have each of them  called one at a time in a row. There's nothing  
magic about this. It's just doing function  composition. We could easily create our own.  
In fact, also the transforms dot compose.  That does the same thing. Yeah, it's going  
to say so we've got a fast. A fast core dot  compose, which as you can see, basically,  
it just says for f in funks x equals f of x.  Yeah, I don't know is there is a yeah, torch.  
Orch vision compose. I think might be the kind of  the old way to do it. Is that right? I'm not sure.  
I have a feeling maybe this is considered the  better way now because it's kind of scriptible.  
I'm not promising that though. But  yeah, that's basically the same thing.  
Okay, so yeah, we can now create a model as  usual. Okay, so basically I copied the get model  
with dropout get drop model from our earlier.  Tiny sorry, our earlier fashion, emnest stuff.  
And I started with kernel size five convolution.  And then yeah, a bunch of rest blocks.  
Yeah, so this is all what  we've used to seeing before.  
And so we can take a look in this  case, as it's quite often seems to   be the case we accidentally end up with  no random erasing. Let's run it again.  
Really doesn't want to do random erasing here  we go. So we can see it. So yeah, that's this  
very small border. You can hardly see sometimes  in a bit of random erasing and it's been done.  
You know, all of the batch is being transformed or  augmented in the same way. Which is kind of okay.  
It's certainly faster. It can be a bit of a  problem if you have like one batch that has lots  
and lots and lots of augmentation being done to  it. And it could be like really hard to recognize.   And that could. Was the loss to be a lot in that  batch. And if you're like been training for ages  
that could. Kind of jump you out of the. You know,  the smooth part of the loss surface. That's that's  
the one downside of this. So I'm not going to  say it's always a good idea to do augmentation at   batch level, but it can certainly speed things  up. A lot if you don't have hints of CPUs.  
All right. So you can use that summary  thing we created. There's your model.  
And yeah, because we're increasing the doubling  the number of channels as we're decreasing the  
grid size. Our number of mega flops per layer  is constant. That's a pretty good sign that   we're using compute throughout. So yeah, then  we can train it with Adam W. Mix precision. And  
our. Or quintetions. So then did the learning rate  finder. And trained it for 25 epochs. And got.  
Nearly 60% 59%. And yeah, this took quite a while  actually to get close to 60% I got to admit.  
It's. And you can see.   The training sets are already up to 91. So  we're kind of on the verge of overfitting.  
Okay. So then I thought all  right. How do we do better.  
And I wanted to have a sense of like how much  better could we get. And I kind of tend to  
like to look at papers with code, which is a site  that shows. Papers with their code and also like  
how good results did they get. So this is the  image classification on Tiny Imagenet. And at  
first I was like pretty disheartened to see all  these like 90% plus things. But as I looked at  
the. And I thought, oh my, to see what was this.  I realized, I realized something. The first thing   as I noticed that these ticks here represent  extra training data. So these are actually.  
Retrained models that are only fine tuned  on Tiny Imagenet. So that's a total treat.   And then I looked more closely at this one. And  actually, these are also using pre train data.  
So papers with code is actually incorrect. And so  the first ones I could see, which I could clearly  
kind of replicate and make sense of. And so the  highest one that I'm confident of is this 72%.  
And so then I kind of wanted  to get a sense of, or how.  
You know, how, how much work is there to get  from like 60% to 70% and how good is this.  
So I opened up the paper.  And so here's Tiny Imagenet.  
And they've got like, they're basically this  paper turns out to be about a new type of mix   up data orientation. This is the normal kind of  mix up. And this is a special kind of mix up. And  
on a resident 18. Yeah, I see they're getting  like 63, 64 or 65 with various different types  
of mix up. And kind of 64 or 65 for their special  one. And then if they use much bigger models than  
we're using, they can get up to 66 ish. So that  kind of maybe think, OK, this classifier is not  
not bad. But there's clearly room to improve it.  And I can't help myself. I always have to try to  
do better. So this is a good opportunity to learn  about a trick that is used in real res nets, which  
is in a real resident. We don't just say. How  many filters or channels or activations per layer.  
And then just go through and do a. You  know, try to con each time. But instead.  
You can also say. The number of. Res blocks.  
Her. Her kind of down sampling layer.  So this would say do three res blocks.   And you know, then down sample or down sample and  then do three res blocks or something like that.  
I'll do three res blocks. The first of which or  the last of which is a down sample. And then two   res blocks. With a down sample and then two res  blocks the down sample. So this has got a total of  
one, two, three, four, five down samples. But it's  got it's rather than having one, two, three, four,  
five. Res blocks. It's going to have three, four,  five, six, seven, eight, nine. Res blocks. So it's  
nearly twice as deep. And so the way we do that  is we just replace the places it was saying. Res  
block with res underscore blocks. And that's just  a sequential. Which goes through the number of.  
Box. And it's a res block. And you can  do it a couple of ways. In this case.  
I said if it's the last one, then make it straight  to otherwise dried one. So it's going to be.  
Down sampling at the end of each set of res  blocks. So that's the anything I changed. I  
changed res, res block to res blocks. And passed  in. The number of blocks, which is this. Okay. So.  
So the number of mega flops is now seven, ten.  Ish, which is more than double. Right. So. Should  
give. Should have more opportunity to learn stuff,  which also could be more opportunity to overfit.  
So again, we do a L. I find. And. So it's a 25  E. And I didn't actually add more augmentation.  
Okay. And that got up to nearly 62.  So that was a good. Improvesment.  
And you know, interestingly.  It's not overfitting more.   It's actually if anything less, which  you know, there's something about its.  
Ability to actually learn this, which is. And  then you can probably get down or something.  
So I thought, yeah, we're nice to train it for  longer. So I decided to add. More augmentation.  
And to do that. I decided to use  something called trivial augment,  
which is not a very well known. And it's not a  very well known. It's a very well known. It's a   very well known. It's a very well known. But it's  a very well known. But it's a very well known.  
And it comes from Frank Hutter's lab.  He's, he's, Frank Hutter is somebody   who consistently creates. Extremely practical  useful improvements. With much less of the.  
Lonsense that we often see from. And so  this one's kind of a bit of a reaction to  
some previous approaches. Such as one called  auto augment, one called Rand augment. They  
might have both come from Google Brain. I'm not  quite sure where they kind of used lots of like.  
You know, many, many thousands of TPU hours.  To like optimize. How every images, you know,  
or how how each set of images is augmented. And  yeah, what these guys did is they said, well,  
what if we don't do that. But we just randomly  pick a different augmentation for each image.  
And that's what they did. They just, they just  said. The algorithm one is the procedure. Pick  
an augmentation. Pick an amount. Do it.  I feel like they're almost kind of like.  
I'm trying to make a point about  writing this algorithm here.  
Yeah, and they basically find this as at  least as good or often better. Actually,   then the incredibly resource intensive ones. The  incredibly resource intensive ones also kind of  
require a different version for every data set.  Which is why they describe this as a tuning free.  
So rather nicely and surprisingly for  me, it's actually built into paytorch.  
So if we go to paytorch's website  and go to trivial augment wide.  
Yeah, you can they show you some  examples of trivial augment wide.  
We can create our own as well. Now the thing  is I found that doing this at a batch level  
worked poorly. And I think the reason is what I  described earlier. I think sometimes it will pick  
a really challenging augmentation  to see on, you know, and it all   totally don't mess up the loss function. And if  every single image in the batch is like that,  
that it all. Shoot it off into the. And  then we can see the distance parts of the  
weight area. Which is a good excuse to me to show  how to do augmentations on a per item level. Now.  
These actually require, or some of them require.  Having a P I L image, the Python imaging library  
image, not a tensor. So I had to change things  around. So we have to import image from P I L.  
And we have to change our to from  X now. And we're going to do the  
augmentations in there instead for the  training set. So for the training set.  
We're going to say well, I'm back for both.  So we're going to pass in something is just do   you want to do augmentations. So for the training  set, we're going to pass all equals true. And for  
the validation set, we won't. So yeah, so we so  image.open is how you create a P I L image object.  
And then if we wanted augmentations, then.  Do these augmentations. And then convert it  
into a tensor. So torch vision has  a dot to tensor. We can then call.  
And then we can normalize it. And actually, I  decided just to use torch visions normalize.  
The minute either it's fine with this one works  well. And then again, if you want augmentation,   then do your render raise. And if you remember  our render raise was designed to kind of use.  
Zero one distributed Gaussian noise. So you  want that to happen after normalization.  
So that's why I do this order. So yeah, so now we  don't need to use the. Batch tripping thing. We're  
just doing it all directly in the data set. So you  can see, you know, you can do data augmentation.  
In very simple ways without almost  any framework help here. In fact,  
we're really not is we're not doing any and  it nothing's coming from a framework really.   It's just yeah, it's just this little  to film DS we made. And so now yeah,  
we just pass that into a data loaders get deals.  And we don't need any augmentation callback.  
All right, so now we can keep improving things   by doing something called pre activation resnets.  So if we go back to our original resident.  
You might recall that the way we did it.  
We have this a conf block, which  consists two. On Volutions.  
Enerro. The second one has no  activation. And to remind you what conf  
is. That we first of all do a conf and then  optionally we do a normalization and then  
optionally we do our activation function. So  we end up and then the second of those has act  
equals none. So basically what this is saying is  go convolution norm activation convolution norm.  
That's what self dot coms is. And then  this is the identity path. This does   nothing at all. If there's no down  sampling or no change of channels.  
And then we apply the activation function, the  final activation function to the whole thing.  
So that was how the original res block  was designed, which is kind of a bit  
of an accident because I. To be honest, when I  wrote that I didn't bother looking at the paper,   I just did whatever seemed reasonable in my  head. But yeah, then looking into it further,  
I looked at this this slightly later paper by  the same author as the resident paper coming her.  
And. And then I was just timing her out drew.  You know, this. This version here on the left,  
as you can see, it's conf norm value, conf norm  add value. And yeah, he basically pointed out.  
Yeah, you know what? Maybe that's not great  because the value is being applied to the  
addition. So it's not really a really an identity  path at all. So it wouldn't have been nice if we   could have a pure identity path. And so to do  that, he proposed reordering things to go norm  
value conf norm value conf add. And so this is  called a pre act or pre activation res block.  
So that means I had to redefine conf  to do norm then act and then conf.  
So my sequential now has the  activation in both places.  
And so yeah, other than that. Oh, and  then of course, there's no activation  
happening in the res block because it's all  happening in the cons. Does that make sense?  
Yeah, makes sense. Yeah. So this is now the  site. This is exactly the same except we now  
need to have an activation and a batch norm  after all those blocks because previously it  
finished with an activation norm and activation.  So it starts with them. So we have to put these  
at the end. It also means we can't start with a  res block anymore because if he started with a res   block, then it would have an activation function  at the start, which would throw away half of our  
data, which would be a bad idea. So you've got  to be a bit careful with some of the details.  
But yeah, so now you can see that each image  is getting its own augmentation. And so this  
one's been shared looks like it's a door  or something. Got to tell what the hell it   is. It's been shared. This one's been moved.  That looks like this one's also been shared.  
And you can also see they've got  different amounts of random rays on them.  
So yeah, so I thought I tried to  change training that for 50 epochs.  
And that got us to 65% which is, you  know, as good as nearly as good as the  
normal mix up things are getting even  on a resident 50s. This is really good.  
So I won't spend time on this, but I just  mentioned I was kind of curious like, I mean,   one of the things I should mention also  is they trained all these for 400 epochs.  
So I was kind of curious what would  happen if we trained it a bit longer.   I wasn't patient enough to train it for  400 epochs, but I thought I could do  
200 epochs. So I just replicated that  last one, but made it 200 epochs.  
And that got us to 67.5. Which, yeah, is  better than any of their non-special mix ups.  
So I think it just goes to show you can get  you know genuinely state of the results.  
So if we use their special mix up that  would be interesting to try as well.   So if we can match their results there. But  you know, we've built all this from scratch.  
We didn't do the data orientation from scratch  because it's not very interesting, but yeah,  
other than that. So I think that's really  cool. So I know that you did some other  
experience with the pre activation. Right.  Yeah. Right. When I saw that when I saw the.  
Preactivation success, I was quite enthusiastic  about it. So I actually thought like, oh,   maybe I should go back and actually use  it everywhere. But for that really enough,  
I think it's weird like it was worse for fashion,  emnest and worse for like less data augmentation.  
I mean, maybe it's not that weird, but because the  idea of when her at our introduced it, they said,  
this is to train deeper models. You know,  there's a there's a more pure identity path.  
And so with that more pure identity path. And so  that that should kind of let the gradients flow  
through it more easily. And so there should be  a smoother surface weight surface loss surface.  
So yeah, I guess it makes sense that you don't  really see the benefits on less deep models.  
The bit I'm surprised. Because it's like, it's the  thing I should be the that sort of justification  
should be true for small and one is made or  it's easy. Yeah, it does, but smaller models.  
I'm going to have a less bumpy surface anyway.  They've just got less dimensions to be bumpy  
on. And there's less more importantly,  they're less deep. So this less room  
for gradients to explode exponentially. So  they're not as sensitive. But yeah, I mean,  
I can see why they don't necessarily help as  much, but I don't have any idea why they were  
worse and they were quite consistently worse.  Yeah, yeah, I find it quite interesting to. Yeah.  
Yeah, it's quite curious. And it's interesting  that when we do these like experiments on things  
that nowadays are considered pretty fundamental  and foundational, all the time discover  
things that everybody seems to have noticed  or written about or there's plenty of room to.  
As a kind of a more experimental researcher  to do experiments and then go like, oh, that's  
interesting and then try and figure out what's  going on. Yeah. I think a lot of researchers go in  
the opposite direction and they try to start with  like theoretical assumptions and then test them.  
Well, I think about it. I feel like maybe  one of the more successful folks in terms  
of people who build stuff that actually  get used a more experimental first maybe.  
Okay, so. We have a five minute break  since we are kind of on the hour.  
Sure. All right, so let's now look at notebook 25  super reds. I've just. I'll be a few things in the  
previous notebook. It's in transforms and our  data sets now, de norm. And our tripping batch.  
And our to from X. Let me show  you using to from batch here.  
Not even using to from batch. Let's get rid of  that because that's just confusing. Okay, so looks  
like we're doing the per. Let's figure this out.  So what are we doing here? So we've got. We've  
got our two data sets. All right, so the goal of  this is we're going to do super resolution, not.  
Um, classification. So let's talk about  what that means. What we're going to do   is the independent variable will be  scaled down to a 32 by 32 pixel. Um,  
image. And the dependent variable  will be the original image.  
Um, and so to do random crop within  a padded image and random flips.  
Both the independent and the dependent variable  needs to have had exactly the same random cropping   and exactly the same flipping. Otherwise, it can't  say, oh, this is how you do super res to go from  
the 32 by 32 to the 64 by 64. But you're like, oh,  it has to be flipped around and moved around. So  
yes, so for this kind of imagery construction  task. Um, you it's important to make sure that  
your augmentation is done in the same way on the  independent, the dependent variable. So that's  
why we've put it into our data set. And so this  is something people often get confused about and  
they don't know how to do it. But it's actually  pretty straightforward. If we do it this way,   we just put it straight in the data set. And  it doesn't require any framework fanciness.  
Now then what I did do is I then added  random erasing. Just to the training  
set. And the reason for that is I wanted to make  the super resolution task a bit more difficult.  
Which means sometimes it doesn't just do super  resolution, but it also has to like replace some   of the deleted pixels with proper pixels. And so  it gives a little bit more to do, you know, which  
can be quite helpful. It's kind of it's a it's  a it's a data augmentation technique and also  
something to give it like. World on opportunity  to learn what the pictures really look like.  
Okay, so with that in case that though these  are going to do the padding random cropping and  
flipping. The training set will also add random  erasing and then we create data loads from those.  
Would it make sense to use the trivial augment  here? The trivial augment, did you say? Yeah.  
Maybe. Yeah, I particularly see a reason not  to if if if well, only if you found that.  
Overfitting was a problem. And if you did do  it, you would do it to both independent and   dependent variables. So yeah, here you can  see an example, the independent variable,  
some of the in this case, all of them  actually have some random arrays that   dependent doesn't. It has to figure out how to  replace that with that. And you can also see  
that this is very blocky. And this is less blocky.  That's because this has been gone down to 32 by  
32 pixels. And this one's still at the 64 by  64. So in fact, once you go down that far, the  
cats lost its eyes entirely. So it's going to be  quite challenging. It's lost its lines entirely.  
So super resolutions, quite a good task to  try to get a model to learn what pictures   look like. Because it has to figure out like  how to draw an eye and how to draw cats,  
whiskers and things like that. Well,  you're an associat of in general, sorry.  
It's just going to point out that the data sets  are also simpler because you don't have to load   the labels. So there's no difference between the  train and the validation now. Good point. Yeah,  
because the label, you know, it's actually  dependent variable is just the picture. And so.  
Okay, so because to FMDS, to FMDS has a to FMX,  which is only applied to the independent variable.  
The independent variable has applied to  it this pair of resize to 30 by 32 by 32.  
And then interpolate. And what that actually  does is it ends up still with a 64 by 64 image,  
but the pixels in that image are all like doubled  up. And so that means that it's still doing super  
resolution, but it's not actually going from  32 by 32 to 64 by 64 by 64. But it's just going   from the 64 by 64 where all of the pixels are  like two by two pixels. And it's just a little  
bit easier because that way, we can certainly  create a Unet that goes from 32 to 64. But if  
you have the input and output image, the same  size, it can make code a little bit simpler.   I originally started doing it by not doing this  interpolate thing. And then I decided I was just  
getting a bit a bit confusing. And there's  no reason not to do it this way, frankly.  
OK, so that's our task. And the idea is that then  if it does a good job of this, you know, you could  
pass 64 by 64 images into it. And hopefully  it might turn them into 128 by 128 images.  
Particularly if you trained it  on a few different resolutions,   you'd expect it to get pretty good at.  And you could even call it multiple times.  
But anyway, for this, I was just kind of doing it  to demonstrate. But we have in previous courses  
trained, you know, bigger ones for longer with  larger images. And they actually do one of the  
interesting things is they tend to not only do  super resolution, but they often make the images   look better. Because the kind of the pixels  it fills in, it kind of fills in with like  
what that image looks like on average, which  tends to kind of like average out imperfections.  
So often these super resolution models actually  improve image quality as well, fun, we enough.  
OK, so let's consider the dumb way to do things.  We've seen a kind of a dumb way to do things  
before, which is an autoencoder. But when we've  got with low expectations here, because we've   done an autoencoder before, it was so bad it  actually inspired us to create the learner,  
if you remember. So that was back in notebook  eight. And so basically what we're going to do  
is we're going to have a model, which looks  a lot like previous models that starts with a  
res block kernel size five. And then it's  got a bunch of res blocks of stray two.  
But then we're going to have an equal number of  up blocks. And what an up block is going to do  
is it's going to sequentially, first of all,  it's going to do an up sampling nearest 2d,   which is actually identical to this. Right,  so it's going to just double all the pixels.  
And then we're going to pass that through a res  block. So it's basically a res block with like  
a stride of a half, if you like, you know,  it's it's it's it's undoing a stride to it's  
up sampling rather than down sampling.  Okay, so and then we'll have an extra  
res block at the end to get it down to  three channels, which is what we need.  
Okay, so we can do our learning  learning learning right finder on that.   And I just try to pretty briefly for five epochs.  So this model is basically trying to take the  
image that we start out, then kind of squeeze  it into, I guess, a small representation of the  
trying to bring that small representation back up  to then the full super. Exactly right to niche and  
we could have done it without any of the stride  to, you know, I guess we could have just had  
a whole bunch of stride one layers. There's  a few reasons not to do it that way though,  
one is obviously just the computation requirements  are very high because the convolution has to  
scan over the image. So when you keep it at 64  by 64, that's a lot of scanning another is that.  
You're never kind of forcing it to learn higher  level abstractions by recognizing how to kind  
of like, you know, use more channels on a  smaller grid size to represent it. Yeah,  
it's like the same reason that we in classifiers,  we don't leave it at stride one the whole time,  
you know, you end up with something that's  inefficient and generally not as good. Exactly,  
yep, thanks for clarifying to niche. Okay, so the  loss goes down and the loss function I'm using is  
just MSC here right so it's how similar is  each pixel to the pixel it's meant to be.  
And so then I can call capture spreads to get  their predictions and the targets and inputs  
or probabilities targets and inputs I can't  quite remember now. So here's our input images.  
So they're pretty low resolution.   And oh, dear, here's our predicted images so  pretty terrible. So why is that well, basically  
it's kind of like the problem we had with our  earlier order encoder, it's really difficult to  
go from like a two by two or four by four or  whatever image into a 64 by 64 image, you know,  
we're asking you to do something that's just  really challenging and so that would require   a much bigger model trained for a much longer  amount of time, I'm sure it's possible.  
And in fact, you know, latent diffusion, as we've  talked about has a model that kind of does exactly  
that. So in our case, there's no need to make  it so complicated, we can actually do something  
dramatically easier, which  is we can create a Unet.  
The Unets were originally developed in 2015  and they were originally developed for medical  
imaging, but they've been used very, very widely  since. And I was involved in medical imaging at  
the time they came out and certainly they quite  quickly got recognized in medical imaging,   they talk a little bit longer to get recognized  elsewhere, but nowadays they're pretty universal.  
And they are used to stable diffusion and  basically some of the details don't matter here,  
this is like the original paper. So let's focus  on the kind of the broad idea, this thing here  
is called that we're going to call it the down  sampling path, so in this case they started with   five seven two by five seven two images. And it  looks like they started with one channel images  
and then they, you know, as we've seen, then they  took them down to 284 by 284 by 128 and then down  
to 140 by 2056 and then at a 68 by 68 by 512 32 by  32 by 102 for. So here's this down sampling path,  
right, and then the up sampling path is  exactly what we've seen before right so we.  
Up sample and have some I mean in the original  thing they didn't use res nets or res blocks,  
they just use cons, but the idea is the same.  But the trick is these extra things across here,  
these errors, which is copy and crop what we can  do is we can take so under in the up sampling  
we've got a 512 by 512 here, sorry a 512 channel  thing here we can up sample to a 512 channel  
thing. We can then put it through a  con to make it into a 256 channel thing  
and then what we can do is we can copy across the  activations from here now and they actually do  
things in a slightly weird way where they're  down sampling they had 136 pixels by 136 and  
over here they have 104 by 104 so they crop  out the center bit. Because of just kind of  
like the slightly weird way they did basically  weren't padding things nowadays we don't have  
to worry about that that cropping so what we do  is we literally copy over this these activations  
and we then either concatenate or add and you  can see in this case they're concatenating see  
how there's the white bit in the blue bit so.  They have concatenated the two lots together  
so actually I think what they did here is they  went from a 52 by 52 by 512 to 104 by 104 by  
256 and I think that's what this little blue  rectangle here is and then they had another.  
Copy copied out the 104 by 104 by 256 and then  put the two together to get a 104 by 104 by 512.  
And so this these activations half are from the  up sampling and half are from the down sampling  
from earlier in this whole process and it might  be easiest to understand why that's interesting  
when we get all the way back up to the top  where we've got this 393 by 392 by 392 thing.  
The thing we're copying across now is just two  convolutions away from the original image so  
like for super resolution for example we wanted to  look a lot like the original image so in this case  
we're actually going to have an entire copy of  almost something very much like the original image   that we can include in these final convolutions.  And so did I hear we have you know something  
that's kind of like the somewhat down sampled  version we can use here and the more down sampled   version we can use here so yeah that's that's  how the Unet works. Do I think you guys have  
anything to add like things that you found this  helpful to understand or anything surprising.  
I guess it's shaping things these days a lot  of people tend to just out so you've got the  
you know the outputs from the down layer are the  same shape the inputs for the corresponding like   up block and then they just kind of add the.  Yeah, particularly for super resolution adding  
might make more sense than concatenating because  you're like literally saying like oh this little   2 by 2 bit is basically the right pixel but it  just have to be slightly modified on the edges.  
Yeah, it also makes me think of like a boosting  sort of thing where if you think about like the  
fact that the location from the original and  just being passed all the way across at that   highest skip connection then the rest of the  network can be effectively producing an update  
to that rather than having to recreate the whole  image it's. Or to put it another way it's like a  
resonant but there's a skip connections  right but the skip connections are like  
jumping from the start to the end and a  bit after the start to a bit before the end   and I guess a resonance a  bit like boosting to. Yeah.  
Yeah, I mean it was kind of good to say so  yeah basically I think it compared to like  
the denoising on encoder where like we saw like  the results were like even worse than I guess the   original image here I guess the worst it can be  is basically the rich of them is you know I guess  
it's just like a similar sort of intuition behind  the the the the resident. And how that works so  
yeah I mean it could be worse if these cons at the  end are incapable of undoing what these cons did  
which is like one argument for maybe  why there should also be a connection   from here over to here and maybe  a few more cons after that which  
is something I'm kind of interested in  and not enough people do in my opinion.  
Another thing to consider is that they've  only got two cons down here but at this point.  
You have the benefit of only being a 28 by 28  you know why not do more computation at this  
point you know so there's a couple of things that.  Sometimes people consider but maybe not enough.  
So let me try to remember what I did. So in my  Unet here so we've got the down sampling path  
which is a list of res blocks now a module list  is just like a sequential except it doesn't  
actually do anything so then in the forward  we have to go through the down path and. The  
X equals LX each time so it's basically yeah it's  a quench with it doesn't actually do anything  
and so the up path is exactly the same  as before it's a bunch of up blocks.  
And then like we saw before the final  ones going to have to go to three channel.   But now for our forward. What we're going  to do is we're going to keep track of.  
Since we're going to be copying this over here  and copying this over here we have to save it  
during the down sampling path so we're going  to save it in a. Something called layers so  
I actually decided to do the little trick I  mentioned which is to save the very first input.  
So I saved the very first input I then  put it through the very first res block.  
And then we go through each  in the downward path. And  
there's actually no need at all for there to  be an I L here doesn't have to be a numerator  
because we don't use I. Okay so we go  through the downward path so for this   L for layer so for each layer in the downward  path append the activations so that again as  
we go through each one we're going to be able  to copy them over by saving them for later.   And then call the layer. Okay so how many layers  if we got there's n layers that we stored away  
so now we're going to go through the up sampling  path and again we're going to call call each one.   But before we do we're going to actually do the  thing that john I mentioned which is rather than  
concatenating unless we're back at unless with  this is the very first layer because the very  
first up sampling layer there's nothing to copy.  Right so this is the very first up sampling layer  
it's just add the saved activations and then  call the layer. And then right at the very end  
we'll add back the very first layer and then  pass it through the very last last. Res block.  
All right, maybe that last one should  be concatenated i'm not sure any who   this is what I did. Now the next thing  that I wondered about was like how to  
initialize this and basically what I wanted to  do is I wanted to initialize this so that when  
it's when it's untrained it would the output  of the model would be identical to the input.  
Because like a reasonable starting point for like  what does this look like so yeah what does this   look like following super resolution would be this  you know that's a reasonable starting point. So I  
just created this little zero weights thing which  zeros out the weights and biases of a layer right  
so I created the model. And then I said okay let's  look at the very end of the up sampling path.  
And we'll call that the last resident and so  let's zero out the very last convolutions.  
And also the ID connection and so that means that  
whatever it does for all this at the very last.  Whatever it does for all this at the very end.   It's going to have. Nothing in there this will  be zero so that means that this will be equal  
to layers zero. And then that means we also want  to make sure that this doesn't change anything.  
So then we can just zero out the weights  there. That's probably not quite right is it.  
I guess I should have actually set  those to like an identity matrix.   Maybe I try to do that later. But at least it's  something that would be very easy for it to.  
I have a question. Jo, I mean yeah. The the  Sarah weights I see a lot of people do a   thing where they instead like multiply by one  e minus three or one e minus four to make the  
weights really small but not completely zero. And  I don't have a good intuition whether it's like,  
you know, in some sense, having everything said to  zero. Fires off some warnings that maybe this is  
going to be like perfectly balanced on some saddle  point or it's not going to have any signal to   work with. Yeah, it's very small but not quite so  around the weights might be better. Yeah, I think  
so or not to much intuition but more empirical  like all both. I don't I don't think it's an  
issue. And I think it comes from like a lot of  people's PhD supervisors and stuff, you know,   come from back in an era when they were doing like  linear aggression with one layer or whatever. And  
in those cases, yeah, for the weights are the  same. Then no learning can happen because every  
weight update is identical. But in this case, all  the previous weights are different. So there's.  
They all have different gradients and  there's definitely nothing to worry about.  
I mean, multiplying it by a small number would  work to like it's not a problem that yeah setting  
it to zeros. And honestly, I, I have to stop  myself from I mean, that's a problem, but I just.  
I always have this natural inclination to  not want to set them to zeros because of   years of being told not to. But there's  no reason that should be a problem.  
All right. So I just was just like, again, like  that Unet code is very concise and it's very,  
very interesting to see the basic ideas,  you're very simple and. Oh, yeah, to see that,  
I guess. Yeah. Yeah, it's helpful. I think we just  get it into a little bit of code, isn't it? Yeah.  
Thanks. That's very simple code too.  Okay, so we do a lot of find and then we  
train. And you can see, but previously  our loss, even after five epochs  
was 207. And in this case, our loss after one  epoch is 086. So it's obviously much easier.  
And we end up at 073. Okay. So we can  take a look. Is our inputs. And there's  
our outputs. So it's actually better rather  than dramatically worse now. So that's good.  
Yeah, so it's actually not  bad at all, I would say.   And this car definitely looks like I think  it's like a little over smooth, you know,  
I think you could say. So if we look at the  other guys eyes, kids eyes still aren't great.  
I can the original is actually got proper  pupils. So yeah, it's definitely not. Recreated  
the original, but you know, given limited compute  and limited data, like the basic idea is not bad.  
I do worry that the poor koala like it didn't have  eyes here, but like it ought to have known there  
should be eyes in a sense, and it didn't create  any. And maybe it should have done a better job   on the eyes. So my feeling is. And this is pretty  common way of thinking about this is that when you  
use means great error, MSC as your loss function  on these kinds of models. You tend to get rather  
blurry results, because if the model's not sure  what to do, this is going to predict kind of   the average, you know. So one good way to fix  that is to use perceptual loss. And I think  
it was John who taught us about perceptual loss  wasn't it when we did the style transfer stuff.  
So perceptual loss is this idea that we could look  it's kind of similar as well to the the fed idea.  
We could look at the some intermediate layer of  a pre trained model and try to make sure that  
our output images have the same. And then we  could look at the features as the real images.  
And in this case, it ought to be saying like the  real image, you know, if we went to kind of midway  
through a resident, it should be saying like  there should be an eye here. And in this case,   this would not represent an eye very well.  So that would should give it some useful  
feedback to improve how it draws an eye here. So  to do perceptual loss, we need a classifier model.  
So I just used the little, I don't know  why I used the 25 epoch one, I guess,   maybe that's all I had trained with at  that time. So I used a 25 epoch model.  
So then, yeah, I just grab a batch of my  validation set and then we can just try it out by  
calling the classifier model. And here I'm doing  it in fp16 just keeping my memory used down.  
Don't think this dot half would be  necessary since I got auto cast, never mind.  
Okay, this is the same code we had before  for the synth sets. So here is our images.  
So what we've got here.  
Just looking at some of them there, we are  not they I mean, qual is so fine. You know,   I wouldn't have picked this as a parking meter.  I wouldn't have picked this as a bow tie.  
So yeah, so basically what this  is doing here is it's. So, um.  
Showing us the predictions. So the predictions  are not amazing. trolley bus that looks right.  
This is weird. It's called this one a neck  brace and this one a basketball that looks   for the connect base the lab or retriever.  It's got right the tractor. It's got right.  
Centipede's right mushrooms right. So, you  know, you can see how classified it's okay,  
but it's not amazing. I think this  was one with like a 60% accuracy.  
But the important thing is it's like it's  got enough features to be able to like do   an okay job. I have no idea what this is. So I'm  pretty sure it's not a goose. Okay, so the model.  
The model was a very simple,  just a bunch of res blocks.  
Three, four, five, and then at the end, we've  got a pooling flattened dropout linear batch.  
So we don't need. Yeah, so what we're going to  do is just to keep things simple. We're just  
going to grab. I think the end of the three  res block. And so simple way to do that is  
we'll just go from range four to the end of the  model and delete those layers. So if we do that.  
And then look at the model again.  You can now see I've got zero one,  
two, three, and that's it. So this model is going  to, yeah, return the kind of the activations after  
the fourth res block. So for the conceptual  losses, I think we talked about you could like  
pick a couple of different places like there's  various ways to do it. This is just the simplest.  
I didn't even have to use hooks or anything. We  can just call sea model. And in fact, if we do it.  
So just to take a look at this looks like and  again, we're going to use. And then we're going to   use a mix precision here. We can grab our Y batch  as before, put it through our classifier model.  
And so now that we've done this, this is now going  to give us those intermediate level features.  
So the features, what's the shape of  them? It's batch size one or two four by  
the number of channels of that layer by  the height and width of that layer. So   we're going to be using the features we're  going to be using for the perceptual loss.  
And so when I was doing this, I kind of  wanted to check whether things were vaguely   looking reasonable. So I would expect  it that these features from the actual Y  
should be similar to if I use our model.  
So something that I did, I thought, okay, if we  took that model that we trained, then we would  
hope that the features were at least of the same  sign from, you know, from the result of the model,  
then they are in the real images. So this  is just me comparing that and it's like,  
oh, yeah, they are generally the same sign.  So this is just a little checks that I was   doing along the way. And then I also thought  I kind of look at the msc loss along the way.  
Yeah, so there's no need to keep all those in  there. It was just stuff I was kind of doing  
to like debug as I went, what not even debug  to identify ahead of time as of any problems.  
So now we can calculate our loss function. So our  loss function is going to be the the msc loss just  
like before between the input and the target,  just just all that's being passed in here. Plus   the msc loss between the features we get out of  sea model and the features we get from the actual.  
And the features we get from the actual target  image and so the features we can calculate  
for the target image now the target image where  not going to be modifying that at all. So we do  
that bit with no gradient. But we do want to  be able to modify the thing that's generating  
our input that the model we're trying to  optimize. So we do have gradient for that.   So in each case we're calling the classifier  model one on the target and one on the input.  
And so those are giving us our features.  Now then we add them together, but they're  
not particularly similar numerically like they're  very different scales and we wouldn't want it to  
focus entirely on one or the other. So I just  ran it for epoch or to check what the losses  
were looked like and I noticed that the feature  loss was about 10 times bigger. So my very hacky   way was just to divide it by 10. But honestly,  like that detail doesn't tend to matter very  
much in my opinion, which there's nothing  wrong with doing it in a rather hacky way.  
There are papers which suggest  more elegant ways to handle it,  
which isn't a bad idea to save you a bit of time  if you're doing a lot of messing around with this.  
Jeremy, I don't know if you know it, but the  new VAD Coda from stability AI for the stable  
diffusion auto encoder. They train some with just  mean squared error and some with mean squared  
error combined with the perceptual loss and they  had a scaling factor of times 0.1. So exactly  
the same. So the answer is 0.1 that's that's the  official. And Andre Capathy says that the correct  
learning rate to use is always 4 reneg 3. So we're  getting all this sorted out now. That's good.  
All right. So for my unit, we're going to  do the same stuff as before in terms of   initializing it. Do our LR find. Train  it for 20 epochs and obviously the loss  
is not comparable because this is lost now  incorporates the perceptual loss as well.   And so this is one of the challenges with these  things is like is it better or worse? Well,  
we just turn it have to take a look and  compare, I guess. Maybe I should copy over   our previous models images. So we can compare.  Okay, there's our inputs. There's our outputs.  
And yeah, look, he's got pupils  now, which he didn't used to have.   Qualistial doesn't quite have eyeballs,  but like it's definitely less.  
You know, out of focusy looking. So  yeah, I'm just looking at going on. Yeah,  
so there's some of them are going to be  flipped because this is copied from earlier.   So yeah, there's clipping and cropping  going on. So there won't be identical.  
Yeah, you can also see like the background.  Like was all just blurred before where else  
now it's got texture, which if we look at the  real, the real has texture, you know, so. So  
yeah, clearly the perceptual losses  improved matters. Quite significantly.  
There's an interesting thing here, which is that  there's not really any metric we can use now,   right? Because if you didn't mean squared error,  the one that's trained me and means good error  
would probably do better, but visually it looks  worse. Yeah, if we use like an if ID, well,   that's based on the features of the pre-trained  network. So that would probably be by it's by  
the one that's trained using those features, the  perceptual loss. And so you get back to this very   old school thing of like, well, actually, how are  you choosing is just looking at evaluating right.  
And when you speak to someone like Jason Antick,  who's made a whole career out of, you know, image   restoration and super resolution and colorization.  That is like a big part of his process, even now  
is still like looking at a bunch of images  to decide where there's something is better.  
Rather than relying on these. Yeah, some  PhD student yelled at me on Twitter a few   weeks ago for like saying like, look, this cool  thing our student made look how don't they look  
better and he was like, don't you know there's  rigorous ways to measure these things. This is   not a rigorous approach at all. It's like,  PhD students, man, they got all the answers.  
I never human looking at a picture and  deciding if they like it or not. That's insane.  
Well, I'm a PhD student, I agree, though  that we should be looking at. So yeah,   okay, some PhD students are better  than others. That's fair enough.  
What's this? Right. Okay. So  talking of trading, let's do that.  
So we're going to do something which is  kind of fast a is favorite trick and has  
been since we first launched, which is gradually  unfreezing pre trained networks. So in a sense,  
it seems a bit funny to initialize all of this  down path randomly because we already have a  
model that's perfectly capable of doing something  useful on Tiny Imagenet images, which is this.  
So yeah, what if we took our unit, right, and  for the model dot start, which to remind you.  
Is the res block right at the front. Why don't we  use the actual weights of the pre trained model.  
And then for each of the bits in the down  sampling path, why don't we use the actual weights  
that we used from that as well. And so this is  a useful way to understand how we can copy over  
weights, which is that any part of a module and  end end module is itself an end end module. Has  
a state dict, which is a thing you can then  call load state dict to put it somewhere else.  
So this is going to fill in the whole res block  called model dot start with the whole res block,  
which is p model zero. So here's how we can copy  across. Yeah, that's starting one and then all the  
down blocks are going to have the rest of it.  This is basically going to copy into our model   rather than having random weights. We're going to  have all the weights from our pre trained model.  
And then since they're. They're good at doing  something. They're not going to do in super  
resolution, but they're going to do something.  Why don't we assume that they're good at doing  
super resolution. So turn off requires  grad. And so what that means if we now  
train it's not going to update any of the  parameters in the down block. I guess I  
should have actually done model model dot start  requires grad as false to now think about it.  
And so this is the classic fine tune approach from  fast AI the library. We're going to do one epoch  
of just the upsampling path. And that gets us to a  loss of two five five now our loss function hasn't  
changed. That's totally comparable. So previously  I won epoch was three eight five. And in fact,  
after one epoch with frozen weights  for the down path, we've beaten  
this now. This is in a sense totally  cheating, but in a sense it's totally  
not it's totally cheating because the  thing we're trying to do is to generate  
for the perceptual loss intermediate  layer activations, which are the same as  
this. And so we're literally using that  to create intermediate layer activations.  
So obviously that's going to work. But why  is it okay to be cheating? Well, because  
that's actually what we want like to be able to do  super resolution. We need something that can like.  
Recognize isn't I here. So we already  has something that know that there's an   eye there. And in fact, interestingly  this thing trained a lot more quickly.  
Than this thing. And it turns out it's better at  super resolution. Than that thing, even though it  
wasn't trained to do super resolution. And I think  that's because the signal, which is just like.  
What is this is a really simple signal to use.  So yeah, so we do that. And then we can basically  
go through and set requires gradicals true  again. And so the basic idea being here that.  
Yeah, when you've got a bunch of random weights,  which is the whole up sampling path and a bunch  
of pre trained weights, the down sampling path.  Don't start then fine cheating the whole thing.  
Because at the start it's going to be crap, you  know, so it's so just train the random weights  
for at least an epoch. And then set everything  to unfrozen. And then we'll do a 20 epochs on  
the whole thing. And so go from 255 to  249. To a 7198. So it's improved a lot.  
So there fight with the. With using these weights,  the in comparing that to the perceptual loss, the  
perceptual loss is looking at the. And the data  sample, they throw the super resolution images,  
as well as we're incorporating the ways  that's for the down sampling path. And so   that's right. And yet I guess the original.  Downgraded. But we are just asking them. So  
if you have zeros in the up sampling path  that it's going to be the same. So it is  
very easy for it to get the correct.  Activations in the up sampling path.  
Yeah, I mean, then it's kind of weird  because it goes all the way back to the top,   create the image and then goes into the  class of C model, the classifier again.  
But I think it's going to create basically  the same activations. It's a bit confusing  
and weird. So yeah, I mean, it's not totally  cheating, but it's some. It's certainly an  
easier problem to solve. Yeah. Okay, so let's  get our results again. So there's our imports.  
Yeah, so that's looking pretty impressive.  So the kid has a, yeah, definitely looks  
pretty reasonable now. How looks pretty  reasonable. We still don't have eyes for  
the koala such as life, but definitely  the background textures look way better.  
The candy store looks less  much better than it did.   Medicine looks a lot better than it did. So  yeah, it's really. I think it looks great.  
So then we can get better still. This  is not part of the original unit, but.  
You know, making better models is often about like  where can we squeeze in law computation, give it  
opportunities to do things and like there's  nothing particularly that says that this down   sampling thing is exactly the right thing you need  here. It's being used for two things. This con and  
one is this kind of, but those are two different  things. And so it's kind of having to like learn   to squeeze both purposes into one thing. So I had  this idea, probably I'm sure lots of people have  
this idea, but whatever. I had this idea, which  is why don't we put some res blocks in here,  
which are called cross connections or cross  cons. So I decided that a cross con is going  
to be just a res block by a con. And  so the Unet I just copied and pasted,  
but now as well as the downs, I've also got  crosses. And so the crosses across cons. So now  
rather than just adding the layer, I add  the cross con. I applied to the layer.  
Yeah, I really should have added a cross con for  this one as well. Now I think about it. This is   probably the one that wants at the most. Well,  other time. Okay, so now again, we can definitely  
compare. So this is one nine eight. So everything  else was the same. So I did the same thing of  
because you know the down sampling is the same. So  we can still copy in the state deck requires grad.  
And it's better. One eight nine quite a lot better  really because you know, this is these are hard  
to get improvements. So if we can notice  anything. Hey, look. It's got an eye just.  
Yeah. So how about that? At this point, it's  almost quite difficult to see whether it's  
an improvement or not, but. I think there's a bit  of an eye on the koala. I think it's encouraging.  
Yeah. So that's our super res. Oh, man.  The bad news is we're out of time. Okay.  
We didn't promise to do diffusion  Unet this lesson. We built a Unet.  
We built a Unet. We did. And it's and we did  super resolution with it and it looks pretty good.  
So, I got to admit, I haven't thought about  like exercises for people to do. What would  
be useful things for people to try with  like maybe they could create a Unet.  
They could learn about segmentation, create  a Unet for segmentation or. Oh, you know,  
there are a couple of lines where you. I was just  going to see there were a couple of ways we said,  
oh, I should have tried this and you're trying  that. I think that's how we see. Yeah, basically.  
I think that's obviously a good next step. I'm  going to say style transfer is a good idea to  
do. I think with a Unet. So style transfer, you  can actually set up a loss function so that you  
can create a Unet that learns to create images  that look like Van Gogh. You know, for example.  
It's a totally different approach. It's a tricky  one. I think I think when I was playing with that,  
it almost helped to not have the skip  connections at the highest resolutions.  
Otherwise, it just really wants to copy the  input and modify it slightly. Interesting.  
Maybe doing. Which one would be better there  too. Oh, yes, that's a good point. Yeah.  
Cool. Well, we'll put some stuff up on the  website about, yeah, you know, ideas. I'm   sure some students, hopefully by the time you  watch this, we'll have some ideas on the forum  
of things I've tried to. Yeah, right. Yeah, the  colorization is nice because it's. Colorization  
at the transform is just. To grace, scaling  back. Oh, yes. And then that's, yeah, that's  
really. Actually, okay. So there's all kinds of  decrappification you could do isn't there. If you   want to keep it a bit more simple, yes, rather  than doing these two lines of code, you could.  
Yeah, just turn it into black and white. It's a  great point. Or you could delete the center every  
time you know, to create like a something that  learns how to fill in. If you delete the left hand  
side and that way that would lead to something  that you can give it a photo in a little invent   a little bit more to the left. Yeah, and then  you could keep running it. Generator panorama.  
Another one you could do would be to like   in memory or something, save it as a really  highly compressed JPEG. And so then you would,  
it would be something that would learn to  remove JPEG artifacts, which then for your like.  
Old photos that you saved with crappy JPEG  compression, you could. Bring them back to life.  
You can probably do like, yeah, you can do like, I  guess, drawing to painting or something like this,   by taking some paintings that might pass and get  through such an edge detection and using that  
as your starting point. Sounds interesting. Oh,  what about watermark removal? You could use PIL  
or whatever to draw. Watermarks, text, whatever  over the top, which is quite useful for like,  
you know, radiology images and stuff sometimes  have personally identifiable information written   on them and you can just like. Learn  to delete it. Yeah, okay, so lots of  
things people can do. That's awesome. Thanks for  your ideas. Basically any image to image task.  
Super. All right. Or just make the super res  better. Try it on full Imagenet, if you like. If  
you've got lots of hard drive space. Thanks, John.  Thanks, Tanishq. See you next time. Thank you.

---

# 24

hi we are here for lesson 24 and once again it's becoming a bit of a tradition
now we're joined by jono and tanishq which is always a pleasure hi jono hi tanishq
hello so are you guys another great lesson yeah you guys looking forward to finally actually completing
stable diffusion at least the unconditional stable diffusion well I should say no even conditional so conditional stable diffusion except for
the clip a bit from scratch we should be able to finish today time permitting oh
that's exciting that is exciting yeah all right let's do it um jump in anytime
you've got things to talk about so we're going to start with a very hopefully named 26 diffusion
unit um and what we're going to do in 26 to
Fusion unit is to do unconditional
diffusion from scratch and um there's not really too many new
pieces if I remember correctly so all you know all the stuff at the start we've we've already seen
um and so when I wrote this it was before I had noticed that the Karas approach was doing less well than the
regular cosine schedule approach so I was still using Keras noiseifly but this is all the same as from the
um the Keras notebook which was 23. um
Okay so we can now create
a unit that is based on
what diffusers has which is in turn based on lots of other prior art
um I mean the code's not at all based on it but the basic the structure is going to be the same as as what you'll get
into fuses um the convolution we're going to use is
the same as the final kind of convolution we used for tiny imagenet which is What's called the
pre-activation convolution so the convolution itself happens at the
end and the normalization and activation happen first so this is a preact
convolution um so then I've got a
unit resnet block so I kind of wrote this before I actually did the
react version of tiny imagenet so I suspect this is actually the same quite possibly exactly the same as the
tiny imagenet one so maybe it's this is nothing specific about this for you net this is just really a pre-act con and a
preact resnet block so we've got the uh the the two comms as per usual and the
identity con now there is one difference though to what we've seen before for resnet blocks which is that this resnet
block has no option to do down sampling no option to do
a stride um this is always dried one
which is our default so the reason for that is that when we get to the thing that
strings a bunch of them together which will be called down block this is
where you have the option to add down sampling and if you do add down sampling we're going to add a stride 2
convolution after the res block and that's because this is how
yeah diffuses and stable diffusion does it um
I haven't studied this closely to Niche Cliff John I don't know if either of you have no like where this idea came from
or why um I'd be curious you know the difference is that normally we would
have the uh have we'd have average pooling here in this connection
um but yeah this different approach is what
a lot of the history of the um diffusers unconditional unit is to be
compatible with the ddpm weights that were released and some follow-on work
from that and I know like then improved gdpm and these others like they all kind of built on that same sort of unit
structure even though it's slightly unconventional if you're coming from like a normal computer vision background
okay architecture came from because like some
of the ideas came from some of the then units but I don't know if ddpm yeah they
had something called efficient unit that was inspired by some prior work that I yeah I can't remember
the like the lineage okay um but anyway yeah the the diffusers one has since become
you know like you can add in parameters to control some of this stuff but yeah it's it's uh we should assume that this
is the optimal approach I suppose um uh but oh yeah I will dig into the history and try and find out how much
like what ablation Studies have been done so for those of you who haven't heard of ablation studies that's where
you like try you know a bunch of different ways of doing things and score which one works better and which one
works less well and kind of create a table with all of those options and so
where you can't find ablation studies for something you're interested in often that means that you know maybe not many
other options were tried because researchers don't have time to try everything
um okay now the unit if we go back to the unet that we used for
um super resolution we'll just go back to our most basic version
um what we did as we went down through the
layers in the down sampling section we stored
um the activations at each point into a list called layers
and then as we went through the up sampling we added those down sampling layers back
into the up sampling activations so that's the kind of basic structure of
a unit you don't have to add you can also concatenate and actually concatenating
is what is um I think it's more common nowadays and I think of the original unit might have
been cat betting although for super resolution just adding seems pretty sensible so we're
going to concatenate but what we're going to do is we're going to try to we're going to kind of exercise our
python muscles a little bit to try to see interesting ways to make some of this a little easier to
at turn different down sampling backbones into units and you also use
that as an opportunity to learn a bit more python so what we're going to do is we're going
to create something called a saved res block and a saved
convolution and so our Downs our down blocks so these are our res blocks containing a certain number of res block
layers followed by this optional straight to conf we're going to use
saved res blocks and saved columns and what these are going to do is going to be the same as a normal convolution and
the same as a normal res Block in terms of normal unit res block but they're going to remember
the activations and the reason for that is that later on in the unit we're going
to go through and grab those saved activations all at once into a big list
so um so then yeah we basically don't have to kind of think about it
and so to do that we create a class called a save module and all save module
does is it calls forward to grab the the res block or con results
and stores that before returning it now that's weird
because hopefully you know by now that super calls the thing in the parent class but
save module doesn't have a parent class so this is what's called a mix-in
and um
it's using something called multiple inheritance um
and mix-ins are as it describes here it's a design
pattern which is to say it's not uh particularly a part of python per se
it's a design pattern that uses multiple inheritance now what multiple inheritance is is where you can say oh
this class called saved res block inherits from two things save module and
unit res block and what that does is it means that all of the methods in both of these will end
up in here now that would be simple enough except we've got a bit of a confusion here
which is that unit res block contains forward and save module contains forward so it's all very well just combining the
methods from both of them but what if they have the same method and the answer is that
the one you list first can call when it calls forward
it's actually calling forward in the later one and that's why it's a mixer it's mixing
this functionality into this functionality so it's a unit res block where we've customized forward
so it calls the existing forward and also saves it okay so you see mix-ins
quite a lot in the python standard library for example the um the the basic
http staff the basic some of the basic thread stuff you know with networking users
multiple inheritance using this mix-in pattern so with this approach then the
actual implementation of Saved res block is nothing at all so pass means don't do anything
so this is just literally just a class which has no in nothing no
implementation of its own other than just to be a mix in of these two classes so a safe convolution is an nn.com2d
with the save module mixed in
um so what's going to happen now is that we can call a saved res block just like a unit
rest Block in a saved conf just like an nnn.com of 2D um but that object is going to end up
with the activations inside the dot saved attribute so now a down sampling block is just a
sequential of Saved rest blocks as per usual the very first one
is going to have the number of n channels to start with and it would always have numbered and an F the number
of filters output and then after that the inputs will be ELC equal to NF because the first one's change the
number of channels and we'll do that for however many layers we have
and then at the end of that process as we discussed we will add to that sequential
a saved conf with straight 2 to do the down sampling if requested so we're going to end up
with a single nn.sequential for a down block um and then an up block
is going to look very similar but instead of using an
nn.com of 2D with a straight 2 up sampling will be done with a sequence of
an up sampling layer and so literally all that does is it just duplicates every pixel four times into a little two
by two grid that's well an up sampling layer does nothing clever and then follow that by
astrid1 convolution so that allows it to
you know adjust some of those pixels is if necessary with a simple three by three con
so that's pretty similar to a stride to down sampling this is
kind of the rough equivalent for up sampling um there are other ways of doing up
sampling this is just the one that's stable diffusion does
so an up block looks a lot like a down block except that now
so as before we're going to create a bunch of unit res blocks these are not saved res blocks of course we want to
use the save results in the up sampling path of the unit so we just use normal
res blocks but what we're going to do now is as we go through each resnet
we're going to call it not just on our activations but we're going to concatenate that with
whatever was stored during the down sampling path so this is going to be a
list of all of the things stored in the downsampling path it'll be passed to the up lock and so
dot pop we'll grab the last one off that list and concatenate it with the activations and pass that to the resnet
so we need to know how many filters there were how many activations
there were in the down sampling path so that's stored here this is the previous number of filters in the down sampling
path and so the res block wanted to add those in
in addition to the normal number
so that's what's going to happen there um
and so yeah do that for each layer as before and then at the end add an up sampling layer if it's been
requested so it's a Boolean uh okay so that's the upsampling block
because that all makes sense so far yeah it looks good okay
okay so the unit now is going to look a lot like our previous unit
um we're going to start out as we tend to with a convolution to now allow us to create a few more
channels and so we're passing to our unit that's just you know how many channels
are in your image and how many channels are in your output image so for normal image normal full color images that'll
be three three how many filters are there for each of those resnet blocks up
blocks and down blocks you've got and in the down sampling how many layers are there in each block
so we go from the column we'll go from in channel so it'll be three to nf0 which this is the number of filters in
the stable diffusion um model uh they're pretty big as you see by
default um and so that's the number of channels we'd create which is like
very redundant in the this is a three by three con so it only contains three by
three by three channels equals 27 inputs and 224 outputs so it's not you know
doing computation useful computation in a sense it's just giving it more space to work with down down the line
which I don't think that makes sense but I haven't played with it enough to be
sure normally we would do like you know like a few res blocks or
something at this level to more gradually increase it because this feels like a lot of wasted effort
um but yeah I haven't studied that closely enough to be sure sorry Jamie just to tweet this is the
default I think the default settings for the unconditional unit in diffusers but the stable diffusion unit actually has
even more channels it has 320 640 and then 1280 1280. cool thanks for
clarifying and it's yeah the unconditional one which is what we're doing right now uh that's a great Point
um okay so then we yeah we go through all of our number of filters
and actually the first res block contains 224 to 224 so that's why it's
kind of keeping track of this stuff and then the second res block is 224 to 448 and then 448 to 672 and then 672 to
8.96. so that's why we're just going to have to keep track of these things um so yeah we add so we have a
sequential for our down blocks and we just add a down block the very last one doesn't have down
sampling which makes sense right because the very last one there's nothing after it so no point down sampling other than
that they all have down sampling and then we have one more res Block in
the middle and which is that the same as what we did okay so
we didn't have a middle res Block in our original unit here um
what about this one do we have any mid locks nope so we haven't done okay but I mean so it's just another res block that
you do after the down sampling um and then we go through the reverse list of filters and go through those and
adding up blocks and then one convolution at the end to turn it from
224 channels to three channels um okay and so the forward then
um is going to store in saved all the layers just like we did
back with this unit but we don't really have to do it
explicitly now we just call the sequential model and thanks to our automatic saving each
of those now will we can just go through each of those and grab their dot saved
so that's Andy we then call that mid block which is just another res block and then same thing okay now for the apps and what we
do is we just passed in those saved right and just remember it's going to pop them out
each time um and then the comp at the end so
that's uh yeah that's it that's our unconditional
model um it's not quite the same as the diffusers unconditional model because it doesn't have attention which is
something we're going to add um next but other than that this is the
same so let's for because we're doing a simpler problem which is fashion mnist we'll use less channels than the default
using two layers per block is standard um one thing to note though is that in
the up sampling blocks it actually is going to be three layers num layers plus one
and the reason for that is that the way stable diffusion and diffusers do it is
that even the output of the down sampling is also saved so if you have num layers
equals two then there'll be two res block saving things here and one con saving things here so you'll have three
saved cross connections so that's why there's an extra plus one here
okay and then we can just train it using mini AI as per usual I didn't save it after I
last trained it sorry about that so trust me it trained um
okay now that oh okay no that is actually missing something else important as well as
attention the other thing that's missing is that thing that we discovered is pretty important which is the time
embedding um so we already know that sampling
doesn't work particularly well without time embedding so I didn't even bother sampling this and I didn't want to add all this stuff necessary to make that
work a bit better I thought let's just go ahead and do time embedding
so time embedding there's a few ways to do it and the way it's done in stable
diffusion is what's called sinusoidal embeddings
the basic idea maybe we'll skip ahead a bit the basic idea is that we're going to
create a res block with embeddings where forward is not just going to get
the activations but it's also going to get t which is a vector that represents the
embeddings of each time step so I'll actually beat it it'll be a matrix because it's really everything in the
batch but for one element of the batch it's a vector and it's an embedding in exactly the same way as when we did NLP
each token had an embedding and so the word the would have an embedding and the
word jono would have an embedding and the word tanishq could have an embedding although tanishq would probably actually
be multiple tokens um until he's famous enough that he's mentioned in nearly every piece of
literature at which point tanishq will get his own token I expect um
that's how you know when you've made it um so the time and betting will be the
same t t of you know time step 0 will have a particular Vector time step one
we'll have a particular vector and so forth um well actually you know we're doing Keras so actually they're not time step
one two three they're they're actually sigmas you know um so they're continuous but same idea
um a specific um value of Sigma which is actually what
T is going to be slightly confusingly um we'll have a specific embedding now
we want two values of Sigma or t which are very close to each other should have similar
embeddings and if they're different to each other they should have different embeddings
um so how do we make that happen you know and also make sure there's a lot of
variety of the embeddings across all the possibilities so the way we do that is with these
these sinusoidal time steps so let's have a look at how they work so you
first have to decide how how big do you want your embeddings to be just like we do at NLP
that does the word the is it represented by eight Floats or 16 Floats or 400
Floats or whatever let's just assume it's 16 now so let's say we're just looking at the a
bunch of time steps which is between negative 10 and 10 and we'll just do 100 of them I mean we don't actually have
negative sigmas or t so it doesn't exactly make sense it doesn't matter it's you know because it shows you the
idea um and so then we say like okay what's the
largest time step you could have all the largest Sigma that you could have interestingly
every single model I've found every single bottle I found uses ten thousand
for this even although that number actually comes from the NLP Transformers literature and
it's based on the idea of like okay what's the maximum sequence length we support you could have up to 10 000
things in it you know in a document or whatever in a sequence
um but we don't actually have a sigmas that go up to ten thousand so I'm using the number that's used in real life in
stable diffusion and all the other models but it's interestingly this here purely as far as I can tell as a
hysterical accident because this is like the maximum sequence length that NLP
Transformers people thought they would need to support okay now
what we're then going to do is we're going to be then doing e to the power of a bunch of things
and so that's going to be our exponent and so our exponent is going to be equal
to log of the period which is about nine times
the numbers between naught and one eight of them because we've got space we
said we want 16. so you'll see why we want eight of them and not 16 in a moment but basically here are the eight
exponents we're going to use so then not surprisingly we do e to the power of that
okay so we do e to the power of that of each of these eight things
and we've also got the actual time steps so imagine these
are the actual time steps we have in our batch so there's a batch of a hundred
and they contain these this range of sigmas or time steps
so to create our embeddings what we do is we do a
outer product of the exponent.x and the time steps
this is step one and so this is using a broadcasting trick
we've seen before we add a unit axis at X is zero here
and add a unit access sorry an axis sorry on access one yeah and at a unit
access on axis zero here so if we multiply those together then it's going
to broadcast this one across this axis
and this one across this axis so we end up with a 100 by 8. so it's basically
you know a Cartesian product or the possible combinations of time step and exploded multiplied together
um and so here's like um you know a few of those different
exponents for a few different values
um um Okay so that's not very interesting
yet we haven't yet reached something where each time step is similar to each
next door time step um you know over here you know these embeddings look very
different to each other and over here they're very similar so what we then do
is we take the sine and the cosine
of those so that is 100 by eight
and that is a hundred by eight and that gives us a hundred by sixteen
um so we concatenate those together
and so that's a little bit hard to wrap your head around so let's take a look so the across the hundred time steps 100
sigmas uh this one here is the first
sine wave and then this one here
is the second sine wave and this one here is the third
and this one here is the fourth and the fifth so you can see as you go up to higher
higher numbers you're basically you know stretching the
sine wave out and then once you get up to
index eight you're back up to the same frequency as this blue one because now we're starting
the cosine rather than sine and cosine is identical to sine it's just shifted across a tiny bit so you can see these
two light blue lines are the same and these two orange lines are the same they're just shifted across I shouldn't
say lines sorry curves um so when we concatenate those all
together we can actually draw a picture of it and so this picture is 100 pixels across
and 16 pixels top to bottom and so if you picked out a particular
point so for example in the middle here for T equals zero ball Sigma equals zero
one column is an embedding all right so the bright represents
higher numbers and the dark represents lower numbers and so you can see every column
looks different even as though the columns next to each other looks similar
so that's called a Time step embedding and this is definitely something you want to
experiment with right like really I've tried to do the plots I thought are used
for to understand this but and Toronto International also had ideas
about plots for these you know which we've shown um but you know the only way to really
understand them is to experiment so then we can put that all into a function where you just say okay well
how many times sorry what are the time steps how many embedding Dimensions do you want what's the maximum period and
then all I did was I just copied and pasted the previous cells and merged them together
um so you can see there's our outer product
and there's our cat of sine and cos um if you end up with a if you have an
odd numbered embedding Dimension you have to Pat it to make it even don't worry about that so here's something that now you can pass in the number of
sorry the actual time steps or sigmas and the number of embedding dimensions and you will get back
something like this it won't be a nice curve because your times your time steps in a batch won't
all be next to each other it's the same the same idea
a little visualization there which goes back to your comment about the max period being super high yeah you said
like okay adjacent ones are somewhat similar because that's what we want but there is some change and but if you look
all of this first 100 some this is like the half of the embeddings look like they don't really
change at all and that's because 50 to 100 on a scale of like zero to ten
thousand you want those to be quite similar because those are still very early in this like super long sequence that these are designed for yeah
um so here actually we've got wasted space yeah so here we've got a better period of a thousand instead and I've
changed the figure size so you can see it better and it's using up a bit more of the space
yeah um or go to Max period of 10.
um and it's actually now this is yeah using it much better
yeah so like based on you know what you're saying John I I agree like it seems like it would be a lot richer
to use these time step embeddings with a suitable Max period or maybe you just
wouldn't need as many embedding Dimensions I guess if you did use something very wasteful like this but
you used lots of embedding dimensions then gonna still capture some useful ones
um yeah thanks jono so yeah um yeah so this is one of these interesting
little insights about things that are buried deep in code which I'm not sure anybody probably much looks at
um okay so let's do a unit with time step
embedding in it so what do you do once you've got like this
column you know embeddings for for each the item of the batch what do you do with it
well there's a few things you can do with it um what stable diffusion does I think is
correct I'm not promising because I remember all these details right um is that they make their embedding
Dimension length twice as big as the
um number of activations and what they what they then do is we
can use chunk to take that and split it into two separate variables so that just
literally just it's the opposite of concatenate specific two separate variables and one of them is added
to the activations and one of them is multiplied by the activations so this is
a scale and a shift um we don't just grab the embeddings as
is though because each layer might want to do each each res block might want to
do different things with them so we have a embedding projection
which is just a linear layer which allows them to be projected so it's projected from the number of embeddings
two two times the number of filters so that that torch.chunk works
um uh we also have an activation function called Celia this is the
activation function that's used in stable diffusion I don't think the details are particularly important
um but it looks basically like um
erectified linear with a slight curvy bit and also known as swish also known as
swish um and it's just equal to x times
sigmoid X and yeah I think um it's like activation
functions don't make a huge difference but it it'll they can make things train a little
better or a little faster and swish has been something that's worked pretty well so people a lot of people using switch
or Sulu I always call it swish let's say I think Celia was actually where it was
originally the or the gallium paper which had Celia was where it originally was kind of invented and maybe people
didn't quite notice and then another paper called it Swish and and everybody called it Swish and then people were like wait that wasn't the original paper
so I guess I should try to call it silly
um okay so yeah other than that it's just a normal res block so we do our first cons
then we do our embedding projection of the activation function of time steps
and so that's going to be applied to every channel
um sorry to every uh pixel height and width so that's why we have to add unit axes on the height and width that it's
going to cause it to broadcast across those two axes do a chunk do the scale and shift then
we're ready for the second con and then we add it to the input with a
additional con one straight one convey if necessary as we've done before if we have to change the number of channels
okay um yeah because I like exercising a python noctrine muscles
I decided to use a second approach now for the down block on the up clock I'm
not saying which one's better or worse we're not going to use um
multiple inheritance anymore but instead we're going to use uh was not even a decorator it's a
function which takes a function uh what we're going to do now is we're going to use one of 2dd and embrace block
directly but we're going to pass them to a function called saved the function called saved is something which is going
to take as input a callable which could be a function or a module or
whatever so in this case it's a module takes an empress blocker account of 2D
and it returns are callable the callable it returns is identical to
the column that's passed into it except that it saves the result says the activations says the
result of the function where does it save it it's going to save it into a
list in the second argument you pass to it which is the the block
so the save function you're going to pass it the module
we're going to grab the forward from it and store that away to remember what it was
and then the function that we want to replace it with call it underscore f going to take some arguments and some
keyword arguments well basically it's just going to call the original modules.forward passing in the arguments and keyword
arguments and we're then going to store the result in something called the
saved attribute inside here um and then we have to return the result
so then we're going to replace the modules forward method with this function
and return the module so that module has now been yeah I said callable actually
it can't be called what has to specifically be a module because with the forward that we're changing
this at wraps is just something which automatically it's from the python standard library is just going to copy
in the documentation and everything from the original forward so that it all looks like nothing's changed
um now where does this dot saved come from I realized now actually we could make this easier and automate it but I forgot
didn't think of this at the time so we have to create the saved here in the down block it actually would
have made more sense I think here for it to have said if the saved attribute doesn't exist then create it
um which would look like this if not has at your clock comma saved
block dot saved equals if you do this then you wouldn't need
this anymore anyway I didn't think of that at the time so let's pretend
that that's not what we do okay so yeah now
um the the down sampling conf and the resnets
both contain saved versions of modules we don't have to do anything to make
that work we just have to call them we can't use sequentials anymore because we have to pass in the time step to the
resonance as well it would be easy enough to create your own sequential
for things with time steps which passes them along um but that's not what we're doing here
um yeah maybe it makes sense for sequential to always pass along all the extra arguments but I don't think that's how
they work um yeah so our up block is basically exactly the same as before except we're
now using Amber as blocks instead um just like before we're going to concatenate
so that's all the same so okay so a unit model
with time embeddings um is going to look if we look at the
forward the thing we're passing into it now is a tuple containing the activations and the
time steps how are the sigmas in our case so split them out
and what we're going to do is we're going to call that time step embedding function we wrote
saying okay these are the time steps and the number of embed the number of time step embeddings we want
is equal to however many we asked for and we're just going to set it equal to the
first number of filters that's all that happens there um and then
we want to give the model the ability then to do whatever it wants with those to make those work the way it wants to
and the easiest smallest way to do that is to create a tiny little MLP so we create a tiny little MLP which is
going to take the time step embeddings and return the actual embeddings to pass into the resnet box so tiny little MLP
is just a linear layer with
thinking here hmm maybe a layer
that's interesting I um bilinear layer by default has an
activation function I'm pretty sure we should have act equals none here should
be a linear layer and then an activation in the linear layer so I think I've got a bug which we will need to try
rerunning
okay uh it won't be the end of the world it just means all the negatives
will be lost here makes it half only half as useful
um that's not great okay and these are the kind of things
like you know as you can see you've got to be super careful of like where do you have activation functions where do you
have batch norms is it pre-activation is it post activation
um it trains even if you make that mistake and in this case probably not too much
performance but often it's like oh you've done something where you accidentally zero it up you know all
except the last few channels of your like upwards of a blog or something like
that the network tries anyway it does its best it uses what it can
um yeah but yeah it's yeah it makes it very difficult make sure you're yeah you're not giving it those handicaps
yeah it's not like making a credit or something and you know that it's not working because it crashes or because
like it doesn't show the username or whatever um
instead you just get like slightly less good results but since you haven't done it correctly in the first place you
don't know it's the less good results yeah there's not really great ways to do this it's really nice if you can have a
existing model to compare to or something like that which is where kackle competitions work
really well actually if somebody's like got a cackle result then you know that's a really good Baseline and you can check
whether yours this is is as good as theirs
all right so yeah that's what this MLP is for
um so the down yeah the down and up blocks are the same as before the convert is the same as before so yeah so
we grab our time step embedding so that's just that outer product um pass through this sinusoidal this
sine and cosine we then pass that through the MLP um
and then um
we um uh call our down sampling passing in those embeddings each time
you know it's kind of interesting that we pass in the embeddings every time in the sense I don't exactly know why we
don't just pass them in at the start and in fact in NLP these kinds of
embeddings I think are generally just passed into the start so this is kind of a curious difference
and I don't know why it's you know if there's been ablation studies or
whatever do you guys know are there like any popular diffusiony or generative models
with time embeddings that don't pass them in or is this pretty Universal
um some of the fancier architectures like um I think recurrent interface networks
and stuff just passing the conditioning oh
I'm actually not sure yeah maybe they do still do it like at every stage I think some of them just taken everything all at once up front and then do a stack of
Transformer blocks or something like that so I don't know if it's Universal but it definitely seems like all the unit style ones have this
um the times they've been in going in at all the different things we should try some ablations to see yeah if it matters
I mean I guess it doesn't matter too much either way um but yeah if you didn't need it at
every step then it would maybe say if you're a bit of compute potentially
um yeah so now the up sampling you're passing in the activations the time step embeddings and those and that list of
Saved activations um so yeah now we have a
non-attention stable diffusion unit so
we can train that um
and we can sample from it using the same I just copied and pasted
all the stuff from the Keras notebook that we had and there we have it this is our first
diffusion from scratch and the so we wrote every piece of code
for this diffusion model yeah I believe so I mean obviously in terms of the optimized Cuda
implementation Temptations of stuff no but yeah we've we've written up
we've written our version of everything here I believe a big milestone I think so yeah and these feds are about the
same as the Feds that we get from the stable diffusion one um they're not particularly higher or
lower they bounce around a bit so it's a little hard to compare but yeah they're basically the same yeah so that's
um that's a that is an exciting step and
um okay yeah that's probably a good time to
have a five minute break um yeah so yeah okay let's have a five
minute break okay
um normally I would say we're back but only some of us are back jono
jono's uh in Internet and electricity in Zimbabwe is not the most reliable thing
and he seems to have disappeared but we expect him to reappear at some point so we will kick on
Jud Ellis and hope that uh Zimbabwe's uh infrastructure sorts
itself out um all right so
we're going to talk about attention we're going to talk about attention for a few reasons reason number one very
pragmatic we said that we would replicate stable diffusion in the stable diffusion unit has attention in it
so we would be lying if we didn't do attention um number two
attention is um one of the two basic building blocks of Transformers a Transformer layer is
attention attached to a one layer MLP we already know how to create a one layer
or one hidden layer MLP so once we learn how to do attention we'll know how to
we'll know how to create Transformer box um
so those are two good reasons um
I'm not including a reason which is our model is going to look a lot better with attention because I actually haven't had
any success seeing any diffusion models I've trained work better with attention
um so just to set your expectations we are going to get it all working
but regardless of whether I use our implementation of attention or the diffusers one it's not actually making
it better um that might be because we need to use
better types of attention than what diffusers has or it might be because it's just a very subtle difference you
only see on bigger images I'm not sure that's something we're still
trying to figure out this is all pretty new and not many people have done kind
of the diffusion the kind of ablation studies necessary to figure these things out so
um yeah so that's just life um anyway so there's lots of good reasons to to know about attention we'll
certainly be using it a lot once we do it LP which will be coming to pretty shortly pretty soon
um and it looks like jono's reappearing as well so that's good
um Okay so let's talk about um attention
um the basic idea of attention is that we have
um you know an image
and we're going to be sliding a convolution kernel across that image
right and and obviously we've got channels as well or filters
and so this also has that okay and as we
bring it across we might be you know we're trying to figure out like what
what activations do we need to create to eventually you know correctly create our outputs
um but the correct answer as to what's here may depend on something that's way
over here and or something that's way over here so for example
if it's a cute little bunny rabbit and this is where its ear is you know and there might be
two different types of bunny rabbit that have different shaped ears will be really nice to be able to see over here
what its other ear looks like for instance with just convolutions that's
challenging it's not impossible we've talked in part one about the receptive field and as you get deeper and deeper
in a conf net the receptive field gets bigger and bigger but it's you know at higher up it
probably can't see the other ear at all so I can't put it into those kind of more texture level layers and later on
you know even though this might be in the receptive field of here
most of the weight you know the vast majority of the activations it's using is the stuff immediately around it
so what attention does is it lets you um take a weighted average of other
pixels around the image regardless of how far away they are
and so in this case for example we might be interested in bringing in at least a
few of the channels of these pixels over here
um the way that attention is done in stable
diffusion is pretty hacky and
known to be sub-optimal um but it's what we're going to
implement because we're implementing stable diffusion and time permitting maybe we'll look at some other options later but the kind of attention we're
going to be doing is 1D attention and it was a tension that was developed for NLP
and NLP is sequences one-dimensional sequences of tokens so to do attention stable diffusion
style we're going to take this image and we're going to flatten out
the pixels so we've got all these pixels we're going to take this row
and put it here and I'm going to take this row I'm going to put it here so I'm just going to flatten the whole
thing out into one big Vector of all the pixels of Row one and then all the pixels are row
two and then all the pixels of the row three or maybe it's column one column two column three I can't remember the three ways or column wise but it's flattened out anywho
um and then it's actually for each image it's
actually you know a matrix which I'm going to draw it a little bit 3D
because we've got the channel Dimension as well
so this is going to be the number across this way is going to be equal to the height times the width
and then the number this way is going to be the number of channels
um okay so how do we decide
yeah which you know bring in these other pixels well what we do is we basically
create a weighted average of all of these pixels
so maybe these ones get a bit of a negative weight
and these ones get a bit of the positive weight and you know these get a weight kind of
somewhere in between and so we're going to have a weighted average and so basically each pixel so
let's say we're doing this pixel here right now is going to equal its original pixel plus so let's call it X Plus
the weighted average so the sum across I mean this is like X
I plus the sum of
um over all the other pixels
so from zero to the height times the width
um some weight times
each pixel
um the whites
they're going to sum to one and so that way the you know the pixel
value scale isn't going to change well that's not
actually quite true it's going to end up potentially twice as big I guess because it's being added to the original pixel
um so attention itself is not with the X Plus but the way it's done in stable
diffusion at least as it's is that the attention is added to the original pixel so yeah now I think about it
I'm not need to think about how this is being scaled anywho um
so the big question is what values to use for the weights
and um the way that we calculate those is we do a um we do a matrix product
and so our um
or a particular pixel we've got
you know the number of channels for that one pixel
um and what we do is we can compare that
to all of the number of channels for all the other pixels
so we've got kind of this is pixel let's say X1
and then we've got pixel number X2
right all those channels we can take the dot product between those two things
and that will tell us how similar they are
and so one way of doing this would be to say like okay well let's take that dot product for every pair of pixels and
that's very easy dot product do because that's just what the matrix product is equal to so if we've got um
H right W base C and then multiply it by
its transpose H by w base
oh sorry I said transpose and then totally failed to do transpose
um
um multiply bytes transpose
that will give us an H by W by H by W Matrix so each pixel or the pixels are
down here and for each pixel as long as these add up to one then we've got to wait for
each pixel and it's easy to make these out up to one we could just take this matrix multiplication
and take the sigmoid over the last dimension
and that makes sorry not sigmoid um man what's wrong with me
soft mix right yep and take the soft Max over the last
dimension and that will give me something that adds
the sum equals one okay now the thing is
it's not just that we want to find the places where they look the same where the channels
are basically the same but we want to find the places where they're like similar in some particular way you know
and so some particular set of channels are similar in one to some different set
of channels in another and so you know in this case we may be looking for the um pointy earedness activations you know
which actually uh represented by you know this this and this you know and we
want to just find those so the way we do that is before we do this matrix product
we first put our Matrix through
um through a projection um so we just basically put our Matrix
through a matrix multiplication um this one so it's the same Matrix
right but we put it through two different projections and so that lets it pick two different kind of sets of
channels to focus on or not focus on before it decides you know oh this pixels similar to this pixel in the way
we care about and then actually we don't even just multiply it then by the original pixels
we also put that through a different projection as well so there's these
different projections well the projection one projection two and projection three and that gives it the
ability to say like oh I want to compare these channels and you know these channels to these channels to find
similarity and based on similarity you then want to pick out these channels right both positive and negative weight
so that's why there's these three different projections and so the projections are
called k U and V those are the projections and so
they're all being passed the same Matrix and because they're all being
passed the same Matrix we call this self attention
okay I know this is I know you guys know this really well but you also know it's really confusing did you have any
you can add change anything else yeah I I like that you introduced this
without resorting to the um let's think of this as queries at all
which I think is yeah yeah there's actually yeah these are actually
short for key query and value even though I personally don't
find those useful Concepts yeah um your note on the scaling you said oh
so we we said it so that the weights sum to one and so then we'd need to worry about like are we doubling the scale of
X yeah um but because of that P3 AKA V that projection that can learn to oh
yeah scale um this thing that's added to X yeah yeah
appropriately and so it's not like just doubling the size of X it's increasing it a little bit which is why we scatter
normalization in between all of these attention layers um but it's not like as bad as it might
be because we have that V projection um yeah that's a good point
um and if this is uh if P3 or it's actually you know the V NH projection is
initialized such that it would have a mean of zero yeah on average you know it should start
out by not messing with our scale um okay so yeah I guess um
I find it easier to think in terms of code um so let's look at the code there's you
know there's actually not much code I think you've got a bit of background
noise too jono maybe yes that's much better thank you
um um so so in terms of code there's
um you know this is one of these things getting everything exactly right and it's not just right I wanted to get it identical to the stable diffusion so we
can say we've made it identical stable diffusion I've actually imported the attention block from diffusers so we can
compare and it is so nice when you've got an existing version of something to compare to to make sure you're getting the same
results um um so we're going to start off by saying
let's say we've got a 16 by 16 pixel image and this is some deeper level of
Activation so it's got 32 channels uh with a batch size of 64. so nchw and I'm
just going to use the random numbers for now but this has the you know reasonable dimensions for an activation inside a
batch size 64 CNN or a diffusion model or unit whatever okay so the first thing we have to do
is to flatten these out because as I said in in 1D attention this is just
ignored um so it's easy to flatten things out you just say dot view and you pass in
the dimensions of the of in this case the three dimensions we want which is 64 32 and everything else minus one means
everything else so x dot shape colon 2 in this case is added you know obviously it'd be easier
just to type 6432 but I'm trying to create something that I can paste into a function later so it's General so that's
the first two elements 64 32 and then the star just inserts them directly in
here so 64 32 minus 1 so 16 by 16. now then
um again because this is all stolen from the NLP world in the NLP World things
are have they call this sequence so I'm going to
call this sequence by which we're in height by width sequence comes before Channel which is often called D or
Dimension so we then transpose those last two dimensions
so we've now got batch by sequence 16 by 16.
by Channel or dimension so and they'd only call this n s d
sequence dimension um Okay so we've got 32
channels so we now need three different projections that go from
32 channels in to 32 channels out so that's just a linear layer okay and just
remember a linear layer is just a matrix multiply plus a bias so there's three of them
and so they're all going to be randomly initialized to different random numbers we're called I'm going to call them
s k s q s v and so we can then they're just callable so we can then pass the exact same thing
into three or three because we're doing self-attention to get back our Keys queries and values
or k q and V I just think of them as k q and V because they're not really Keys queries and values to me
um so then we have to do the matrix model play by the transpose and so then for every one of the 64
items in the um batch for every one of the 256 pixels there
are now 256 weights so at least there would be if we're done soft Max which we haven't yet so we can now put that into
a self-attention as as jono mentioned we want to make sure that we normalize things so we can proper normalization
here we talked about group Norm back when we talked about batch Norm so group Norm is just batch Norm which has been
split into a bunch of sets of channels um
um okay so then we are going to create our kqv yep John
I was just going to ask should those be just bias equals false so that they're only a matrix multiplied to like
strictly match the traditional implementation
no because okay they also do it that way
yeah they have bias in their attention box cool
um Okay so we've got our qk and V self.q yourself.cater self.v being our
projections um and so uh to do 2D self-attention we
need to find the nchw from your shape we can do our normalization
we then do our flattening as discussed we then transpose the last two
Dimensions we then create our qkv by doing the projections and we then do
a matrix multiply now we've got to be a bit careful now because as a result of that Matrix multiply we've changed the
scale by by multiplying and adding all those things together so if we then simply divide by the
square root of the number of filters it turns out that's you can convince
yourself of this if you wish to but that's going to return it to the original scale
we can now do the soft Max across the last dimension and then
multiply each of them by V so using Matrix multiplayer to do them
all in one go um we didn't mention but we then do one
final projection again just to give it the opportunity to map things you know
to to some different scale you know shift it also if necessary
uh transpose the last two back to where they started from and then reshape it back to where it started from and then
add it remember I said it's going to be X Plus add it back to the original so this is actually kind of self-attention
res resnet style if you like diffusers if I remember correctly does include the
X Plus in theirs but um some implementations like for example pytorch implementation doesn't
okay so that's a self-attention module and all you need to do is tell it how many channels to do attention on
um so and you need to tell it that because that's what we need for our four different projections and our group
Norm better scale um I guess strictly speaking it doesn't
have to be stored here you could calculate it here um but anyway it's either way it's fine
okay so if we create a self-attention layer we can then call it on our
it'll randomly generated numbers and it doesn't change the shape because we
transpose it back and reshape it back but we can see that's basically worked we can see it creates some numbers how
do we know if they're right well we could create a diffuser's attention block
that will randomly generate a q k v projection
sorry actually they've got something else they've got a query key value projection detention and group Norm we
call it qkv progenom they're the same things and so then we can just um zip
those tuples together so that's going to take each pair of first pair second pair third pair and copy
the weight and the bias from their attention block
sorry from our attention block to the diffuser's attention block and then we
can check that they give the same value which you can see they do so this shows
us that our attention block is the same as the diffuses retention block which is nice
um here's a trick which
um neither diffusers nor Pi torch used for reasons I don't understand which is
that we don't actually need three separate attention three separate projections here we could create one
projection from Ni to ni times three that's basically doing three projections
so we could call this qkv and so that gives us 64 by 256 by 96
instead of 64 by 256 by 32.
because it's a three sets and then we can use chunk which we saw earlier so split that into three separate
variables along the last Dimension to give us our q k v and we can then do the same thing Q at Q
dot transpose Etc so here's another version of attention where we just have one
projection for qkv and we chunkify it into separate q k and V and
this does the same thing which is a bit more concise entropy faster as well at least if
you're not using some kind of xla compiler or onx or Triton or whatever
for normal Pi torch this should be faster because it's doing less back and forth between the CPU and the GPU
um all right so that's basic self-attention
um this is not what's done um
basically ever however um because in fact
the kind of question of like or which pixels do I care about
depends on which channels you're referring to you know because like
the ones which are about like oh what color is its ear as opposed to how
pointy is at Sierra might depend more on like what you know is this bunny in the shade or
in the Sun and so maybe you know you mainly kind of want to look at its body over here
to decide what color to make them rather than how pointy them they could um and so
yeah different different channels need to bring in information
from different parts of the picture depending on which channel we're talking about and so the way we do that
is with multi-headed attention and multi-headed Ascension actually turns
out to be really simple and conceptually it's also really simple what we do is we
say let's come back to when we look at C here and let's split them
into four separate
vectors one two [Music] three
four let's split them right and let's do
the whole you know dot product thing on just
you know here's the first part with the first part and then do the
whole dot product part with the second part with the second part
and so forth right so we're just going to do it separately separate matrix model placed for
different groups of channels um and the reason we do that is it then
allows yeah different Parts different sets of channels to pull
in different parts of the image and so these different
groups are called heads
and I don't know why but they are does that seem reasonable
anything to add to that it's maybe worth thinking about why with
just a single head specifically the soft Max starts to come into play
um because you know we said oh it's like a weighted sum just able to bring in information from different parts and whatever else but with soft Max what
tends to happen is whatever weight is highest gets scaled up quite dramatically and so it's like almost
like focused on just that one thing and then yeah like as you said Jeremy like
different channels might want to refer to different things and you know just having this one like single weight
um that's across all the channels means that that signal is going to be like focused on maybe only one or two things
as opposed to being able to bring in lots of different kinds of information based on the different channels right
and you're pointing out I was gonna measure the same thing actually that's a good point so so you're
mentioning the second interesting important point about soft Max you know point one is that it creates something
that's to one but point two is that because of its e to this it tends to
highlight one thing very strongly and yes so if we had single headed attention
your point guys I guess is that you're saying it would end up basically picking nearly all one pixel which would not be
very interesting okay awesome um
oh I see why everything's got thick I've accidentally turned it into a marker and correct
okay so Maori headed attention um I'll come back to the details about
implemented in terms of um but I'm just going to mention the basic idea this is
multi-headed attention and this is identical to before except I've just stored one
more thing which is how many heads do you want um
and then the forward is actually nearly all the same so this is identical identical identical this is new
identical identical identical new identical identical so there's just
two new lines of code which might be surprising but that's all we needed to make this work
and they're also a pretty wacky interesting new lines of code to look at conceptually what these two lines of
code do is they first they do the projection right and then
um
they um they basically take um
the number of heads so we're going to do four heads we've got 32 channels forehead so each head's going to contain
eight channels and they basically grab
um they're gonna we're gonna keep it as being eight channels not as 32 channels and we're going to make each batch four
times bigger right because the the the images in a batch don't
uh combine with each other at all they're totally separate so instead of having one image
containing um 32 channels we're going to turn that
into four Images containing eight channels and that's actually all we need right
because remember I told you that each group of channels each head
we want to have nothing to do with each other so if we literally turn them into different images
then they can't have anything to do with each other because batches don't enter don't react to each other at all
so Ria these rearrate this rearrange and I'll explain how this works in a moment but it's basically saying
think of the channel Dimension as being of H groups of D and rearrange it so
instead the batch channel is n groups of H and the channels is now just D so that
would be eight instead of four by eight and then we do everything
else exactly the same way as usual but now that group that the channels have been
split into groups of H to groups of four and then after that okay well we were thinking of the
batches as being of size n by H let's now think of the channels as being of size H by D that's what these
rearrangers do so let me explain how these work in the diffusers code
um I've uh I can't remember if I duplicated it or just inspired by it they've got things called heads to batch
and batch to heads which do exactly these things and so for heads to batch they say okay
you've got 64 per batch by 256 pixels by 32 channels
okay let's reshape it so you've got 64 images by 256 pixels by
4 heads by the rest so that would be
uh 32 over 8.
channels so it's splitted out into a separate dimension and then if we transpose these two
Dimensions it'll then be n by 4 so n by heads but yeah so by minus 1. and so
then we can reshape so those first two Dimensions get combined into one
so that's what heads to batch does and batch to heads does the exact opposite right reshapes to bring the batch back
to here and then heads by S L by D and then transpose it back again and reshape it back again so that the heads
gets it um so this is kind of how to do it using just traditional Pi torch
methods that we've seen before but I wanted to show you guys this new ish Library called inops
inspired as it suggests by Einstein summation notation but it's absolutely not Einstein summation notation it's
something different and the main thing it has is this thing called rearrange and rearrange is kind of like a Nifty
rethinking of Einstein's summation notation as a cancer rearrangement
notation and so we've got a tensor called T we created
earlier 64 by 256 by 32. and what uh
ionops rearrange does is you pass it this specifications string that says turn this
into this okay this
says that I have a rank 3 tensor three
dimensions three axes containing the First Dimension [Music]
um this is of length n the second dimension is of length s the
third dimension is in parentheses is of length H times d
where H is 8. okay and then I want you to just move
things around so so that nothing is like broken you know so everything's shifted
correctly into the right spots so that we now have each each batch
is now instead n times eight n times h
that a sequence length is the same and D is now the number of channels
previously the number of channels was H by D now it's D so the number of channels have been reduced by a factor
of eight and you can see it here it's turned T from something of 64 by 256 by 32 into
something of size 64 times 8. by 256
by 32 divided by 8. and so this is like really nice because
you know a this one line of code to me is clearer and easier and I liked
writing it better than these lines of code but whereas particularly nice is when I had to go
the opposite direction I literally took this cut it
put it here and put the arrow in the middle like it's literally backwards which is really nice right
because we're just rearranging it in the other order and so if we rearrange in the other order we take our 512 or 256 by 4 thing
that we just created and end up with a 64 by 256 by 32 thing which
we started with and we can confirm that the end thing equals
or every element equals the first thing so that shows me that my rearrangement
is returned its original correctly yeah so I've already had an attention
I've already shown you it's the same thing as before but pulling everything out into the batch
for each head and then pulling the heads back into the channels
so we can do multi-headed attention with 32 channels and four heads
and check that all looks okay so Pi torch has that all built in it's
called nn.ulty at attention be very careful be more careful than me
in fact because I keep forgetting that it actually expects the batch to be the second dimension so make sure you write
batch first equals true to make batch the First Dimension and that way it'll be the same as
diffusers I mean it might not be identical but the same it should be almost the same
the same idea and to make it um self-attention you've got to pass in three things right so the three things
will all be the same for self-attention this is the thing that's going to be passed through the
um Q projection the K projection and the V projection and you can pass different
things to those if you pass different things to those you'll get something called cross attention rather than
self-attention which I'm not sure we're going to talk about until we do it in NLP
um just on the rearrange thing I know like if you've been doing pure pie torch and you're used to like you really know
what transpose and you know reshaping whatever do then it can be a little bit weird to see this new notation but once
you get into it it's really really nice and if you look at the soft retention multi-headed implementation there you've
got Dot View and Dot transpose and Dot reshape uh it's quite fun practice like
if you're just saying oh this inops thing looks really useful like taking an existing implementation like this and
say oh maybe instead of like can I do it instead of dot reshape or whatever can I start replacing these individual
operations with the equivalent like rearrange call and then checking that the outputs are the same like that's
that's what what helped it like click for me was oh okay like I can start to express if it's just transpose then
that's a rearrange with the last two channels I only just started using this and I've
obviously had many years of using reshape transpose Etc in piano
tensorflow Keras Pi torch APL and I would say within
10 minutes I was like oh I like this much better you know like it's
fine for me at least it didn't take too long to be convinced it's not part of um play torch
or anything you've got a pip install it by the way
and it seems to becoming super popular now at least in the kind of diffusion research crowd everybody seems to be
using iron Ops suddenly even though it's been around for a few years
um and I actually put in a issue there and ask them to add in Einstein summation notation as well
which they've now done so it's kind of like your one place for everything which is great and it also works across
tensorflow and other libraries as well which is nice
okay so we can now add that to our um
unit uh so this is basically a copy of the previous notebook except what I've now
done is I've uh this I did this at the point where it's like oh yeah it turns out the cosine scheduling is better so I I'm back to
cosine schedule now this is copied from the cosine schedule book and we're still doing the minus 0.5 thing because we
love it um and so this time I actually decided to
export stuff into a mini AI dot diffusion so so this point I still feel like
things are working pretty well and so I renamed you netconf to pre-con since it's a better name time step embedding
has been exported up sample's been exported this is like a preact
linear version exported um I tried using an end.body head attention
and it didn't work very well for some reason so I haven't figured out why that is yet
um so I'm using yeah this self-attention which we just
talked about multi-headed self-attention um you know the just the scale we have
to divide the number of channels by the number of heads because the effective number of heads is you
know divided across n heads um
and instead of specifying enhance here you specify attention channels so yeah that's what an is 32 attention channels
is eight then you calculate yeah that's what diffusers does I think it's not what NN model head attention does
and actually I think ni divided by ni divided by attention chance is actually
just equal to attention chance so I could have just put that probably
um anyway never mind um yeah so okay so that's all copied in
from the previous one the only thing that's different here is I haven't got the
dot view minus one thing here so this is a 1D self-attention and then 2D
self-attention just adds The Dot View before we call forward and then
reshape it back again um
so yeah so I've got 1D and 2D self-attention okay so now our M res block has one extra thing you can pass
in which is attention channels um and so if you pass in attention
channels we're going to create something called self.attention which is a self-attention 2d layer with the right
number of filters and the requested number of channels and so this is all identical to what
we've seen before except if we've got attention then we add it oh yeah and the attention
that I did here is the non-res netty version so we have to do X Plus because that's more flexible you can then choose
to have it or not have it this way um okay so that's an Embrace block with
a tension um and so now our down block you have to
tell it how many attention channels you want because the res the res blocks need that the app lock you have to know how
many attention channels you want because again the res blocks need that and so now the unit model
where does the attention go okay we have to say how many attention channels you want
and then you say which index block do you start adding attention
so why don't we so then what happens is the attention is done
um here we each resnet
has a tension and so as we discussed you just do the normal res and then the attention right
and if you um
what that in at the very start right let's say you've got a 256 by 256 image
256 by 256 image
then you're going to end up with this Matrix here
is going to be 256 by 2556 on one side and 256 by 256 on the other side
and contain however many you know NF channels that's huge
um and you have to back prop through it so you have to store all that to allow
bat prop to happen it's going to explode your memory so what happens is basically nobody puts
attention in the first layers so that's why I've added a attention
start which is like at which block do we start adding attention and it's not zero
so the reason we just discussed um
another way you could do this is to say like at what grid size should you start adding attention and so generally
speaking people say at when you get to 16 by 16 that's a good time to start adding
attention although stable diffusion adds it at 32 by 32
because remember they're using latents which we'll see very shortly I guess in the next lesson so it starts at 64 by 64
and then they add attention to 32 by 32 so we're again we're replicating stable diffusion here say what a fusion uses
attention start at index one so we you know when we go self.downs.pand we the down block
has zero attention channels if we're not up to that block yet
and uh ditto on the up block except we have to count from the end to our blocks
um now I think about it that should have a
tension as well the mid plug so that's missing
um yeah so the forward actually doesn't change at all for attention
it's only the init um yeah so we can train that
and so previously yeah we got um
without attention we got to 137
and with attention oh we can't compare directly because
we've changed from Keras to cosine um we can compare the sampling though
um so we're getting what are we getting four five five
it's very hard to tell if it's any better or not because
um well again you know a cosine schedule is better but yeah when I've done kind of
direct like with like I've haven't managed to find any obvious improvements
from adding attention but I mean it's doing fine you know four is great
um yeah um all right so then finally did you
guys want to add anything before we go on to a conditional model I was just gonna make a note that um
like I guess just to clarify the with it for the attention part of the motivation was certainly to do this sort of spatial
mixing and kind of like yeah to get it from different parts of the image and mix it but then the problem is if it's
too early where you do have one of you know the more individual pixels
um then the memory is very high so it seems like you have to get that balance of where you don't you kind of want it
to be early so you can do some of that mixing but you don't want to be too early where then the memory usage is is
too high so it seems like there is certainly kind of the balance of trying to find maybe that right place where to
add attention into your network so I just thought I was just thinking about that and maybe that's a point worth
noting yeah for sure there is a trick which is like what they do in for example Vision Transformers or
um the dit that diffusion diffusion with Transformers um which is that if you take like a
eight by eight patch of the image and you flatten that all out or you run that
through some like convolutional thing to turn it into a one by one by some larger number of channels like you can reduce
the spatial Dimension by increasing the the number of channels um and that gets you down to like a
manageable size where you can then start doing attention as well so that's another trick is like patching where you take a patch right right as
um some number like some embedding Dimension or whatever you like to think of it but as a one by one rather than an eight by eight or a 16 by 16 and and so
that's how like you'll see you know a 32 by 32 patch models like some of the
smaller clip models or 14 by 14 patch some of the larger like vit you know
classification models and things like that so that's another thing yeah I guess that's the yeah that's mainly used
when you have like a full Transformer network uh I guess and then this is one where we have that sort of incorporating
the attention into a convolutional uh Network so there's certainly I guess yeah for different sorts of networks
different tricks but yeah yeah and um I haven't decided yet if
we're going to look at vit or not um maybe we should based on what you're
describing I was just going to mention though that um since you mentioned Transformers
um we we've actually now got everything we need to create a Transformer is here's a
Transformer block within with embeddings and a Transformer block with embeddings is exactly the same embeddings that
we've seen before and then we add attention as we've seen before there's a scale and shift
and then we pass it through an MLP which is just
a linear layer an activation or normalize and a linear layer for
whatever reason this is you know gel U which is just another activation function is what people always use in
Transformers um uh for reasons I suspect don't quite
make sense in Vision everybody uses layer norm and again I was just trying to replicate an existing paper but this is just a standard MLP so if you do
um so in fact if we get rid of the embeddings just to show you a true pure Transformer
okay here's a pure Transformer log right so it's just normalize attention add
normalize multi-layer perceptron add that's all a Transformer block is and
then what's a Transformer Network a Transformer network is a sequential of Transformers and so in this diffusion
model I replaced my mid block with a list of your attention
is going to be sequential Transformer blocks so that is a Transformer Network
and to prove it I then reply oh this is another version
which I replaced that entire thing with the pi torch
Transformer Transformers encoder which is called encoder this is just taken from playtorch and so that's
that's the encoder and I just replaced it with that um so yeah we've now built
Transformers now okay why aren't we using them right now and why did I just say I'm not even sure
if we're going to do vit which is Vision Transformers the reason is that um Transformers
you know they're doing something very interesting right which is
um remember we're just doing 1D versions here right so
Transformers are taking something where we've got a sequence right which in our case is
pixels height by width but just call it is a sequence and Eric tree everything in that
sequence has a bunch of channels okay for Dimensions d right
um I'm not going to draw them all but you get the idea um and so for each element of that
sequence which in our case it's you know it's just some particular pixel right and these are just the
filters channels activations whatever activations I guess um
what we're doing is the we first do attention which you know remember
there's a projection for each so so like it's mixing the channels a little bit but just putting that aside the main
thing it's doing is each row
is getting mixed together you know into a weighted average
um and then after we do that we put the whole thing through a multi-layer perceptron and what the
multi-layer perceptron does is it entirely
looks at each pixel on its own so let's say this one
right and puts that through millennia activation Norm
linear which we call an MLP
and so a Transformer network is a bunch of Transformer layers so it's basically
going attention MLP attention MLP
tension Etc et cetera MLP that's all it's doing
um and so in other words it's mixing together
the pixels or sequences and then it's mixing together the channels then it's
mixing together the sequences and the mixing together channels and it's repeating this over and over because of the projections being done
um in the attention it's not just mixing the
pixels but it's kind of it's it's largely mixing pixels and so this
combination is very very very flexible and it's flexible enough
that it provably can actually approximate any convolution that you can
think of given enough layers and enough time and
learning the right parameters um the problem is that for this to
approximate a combination requires an a lot of data and a lot of layers and a lot of
parameters and a lot of compute so if you try to use this so this is a
Transformer Network Transformer architecture if you
pass images into this so pass an image in
and try to predict say from imagenet the class of the image so use SGD to try and
find weights for these um attention projections and MLPs
if you do that on imagenet you will end up with something that does indeed predict the class of each image but it
does it poorly now it doesn't do it poorly because it's not capable of approximating a
convolution it does it poorly because imagenet the entire imagenet as an imagenet 1K is not big enough
to for for a Transformer to learn how to do this however
if you pass it a much bigger data set many times larger than imagenet 1K
then it will learn to approximate this very well and in fact it'll figure out a
way of doing something like convolutions that are actually better than convolutions and so if you then take that
so that's going to be called a vision Transformer or vat that's been pre-trained on a data set
much bigger than imagenet and then you fine-tune it on imagenet you will end up with something
that is actually better than resnet and the reason it's better than resnet
is because these combinations right
which together when combined can approximate a convolution these
Transformers you know convolutions are our best guess
as to like a good way to kind of represent the calculations we should do on images
but there's actually much more sophisticated things you could do you know if you're a computer and you could
figure these things out better than a human can and so a vat actually figures out things that are even better than
convolutions and so when you fine-tune imagenet using
a very you know a vit that's been pre-trained on lots of data then that's why it ends up being better than
a resnet so um that's why you know the the the
um the things I'm showing you are not the things that contain Transformers and diffusion because
to make that work would require pre-training on a really really large data set for a really really long amount
of time um so anyway um so we're we might only come to
Transformers well not in a very long time but when we do do them in NLP in in vision
maybe we'll cover them briefly you know they're very interesting to use as pre-trained models the main thing to
know about them is yeah a vit you know which is a really successful
and when pre-trained on lots of data which they all are nowadays is a very successful architecture but like
literally the vit paper says oh we wondered what would happen if we take a totally plain 1D Transformer
you know and convert it and use convert make it work on images with as few changes as possible
so everything we've learned about attention today and MLPs applies
directly because they haven't changed anything um and so one of the things you might realize that means is that you can't use
a vit that was trained on 224 by 224 pixel images
on 128 by 128 pixel images because you know all of these self-attention
things are the wrong size you know um
um and specifically the the problem is actually the um actually it's not really the attention let me let me take that
back the all of the um uh the
position embeddings are the wrong size and so actually that's something I sorry I forgot to mention is that
um in Transformers the first thing you do is you always take
uh your you know these pixels and you add to them
um a positional embedding and that's done I mean they can be done
lots of different ways but the most popular way is identical to what we did for the time step embedding it's a sinusoidal
embedding and so that's specific you know to
how many how many pixels there are in your image so
um yeah that's an example of one of the things that makes vits a little tricky anyway but hopefully yeah you get the
idea that we've got all the pieces that we need
um okay so with that
discussion I think that's officially taken us over time so maybe we should do the
conditional next time
do you know what actually it's tiny let's just quickly do it now you guys got time
yeah okay so let's just yeah let's finish by doing a conditional model so for a conditional
model um we're going to basically say I want something where I can say draw me the
number sorry draw me a shirt or Draw me some pants or Draw me some sandals
so we're going to pick one of the 10 fashion mnist classes and
and getting you know create an image of a particular class um to do that
um we need to know what class each thing is
now we already know what class each thing is because it's the the Y label
which uh way back in the beginning of time we set
okay it's just called the label so that tells you what category it is
[Music] um so we're going to change our collation
function so we call noiseifiers per usual that gives us our noised image our
time step and our noise um
but we're also going to then add to that Tuple
what kind of fashion item is this and so the first Tuple will be noise
damage noise and label and then the dependent variable as per usual is the noise
and so what's going to happen now when we call our unit which is now a conditioned unit model
as the input is now going to contain not just the activations and the time
step but it's also going to contain the label okay that label will be a number between
0 and 9. so how do we convert the number between 0 and 9 into a vector which represents
that number well we know exactly how to do that and then dot embedding okay so we did
that Lots in part one so let's make it exactly
um you know the same size as our
um time embedding so in number of
number of uh activations in the embedding it's going to be the same as our time
step embedding and so that's convenient so now in the forward we do our time step embedding as usual
we'll pass the labels into our conditioned embedding
the time embedding we'll put through the embedding letter P in before and then we're just going to add them together
that's it right so this now represents a combination of the time and the fashion
item Plus and then everything else is identical in in both parts so all we've added is
this one thing and then we just literally sum it up so we've now got a
joint embedding representing two things and then yeah and then we train it
um and you know interestingly it looks like the loss well it ends up about the same
but it's you know you don't often see 0.03 ones it's you know it's it's it is a bit easier for it to do a conditional
embedding model because you're telling it what it is just it makes it a bit easier so then to
do um conditional sampling you have to pass in what
type of thing do you want um of from these labels
um and so then we
um create a vector just containing that number repeated
however many times there are in the batch and we pass it
to our model so our model has now learned how to denoise something of type c and so now if we say like oh trust me
this noise contains is a noise damage of type c it should hopefully denoise it
into something of type c um that's that's all there is to it there's no
magic there um so yeah that's all we have to do to change the sampling um so like we didn't have to change ddim
Step At All right literally all we did was we added this one line of code and we added it
there um so now we can say okay let's say class ID 0 which is t-shirt top
so we'll pass that to sample and there we go well everything looks
like t-shirts and tops um
yeah okay I'm glad we didn't leave that till next time because it's we can now say
we have successfully replicated everything in stable diffusion
except for being able to create whole sentences which is what we do with clip
getting really close yes well except the clip requires as well
I guess we'll we might do that next or depending on how research goes
all right we still need the latent diffusion part oh good point lightness okay we'll
definitely do that next time so let's see yeah so we'll do a vae
and latent diffusion um which isn't enough for one lesson so
maybe some of the research I'm doing will end up in the next lesson as well but yes okay thanks for the reminder
um although we've already kind of done Auto encoders so vaes are going to be pretty
pretty easy well thank you tanishq and jono
fantastic comments as always glad your internet slash power reappeared jono
backup
all right thanks gang cool thank you that was great

---

# 25

hi everybody and welcome to the last lesson of part two greetings jono and
greetings Tunisia how are you guys doing good thanks doing well excited for the last lesson
it's been an interesting fun Journey yeah I should explain
we're not quite completing all of stable diffusion in this part of the course
there's going to be one piece left for the next part of the course which is the clip embeddings and that's because clip
is NLP and so the next part of the course we will be looking at NLP so we
will end up finishing stable diffusion from scratch but we're going to have to have a
significant diversion and what we thought was given everything that's happened with um
gpt4 and stuff since we started this course we thought it makes more sense to delve into that quite deeply more more
more soon um and delay clip as a result so
hopefully people there feel comfortable with that decision but I think we'll have a lot of yeah exciting NLP material coming up so
that's the rough plan um all right so [Music] um
I think what we might do is maybe start by looking at a really interesting and
quite successful application of pixel level diffusion by applying it not to
pixels that represent an image that pixels that represent a sound which is pretty crazy so maybe jono of course
it's going to be jono he does the crazy stuff which is great so jono show your your crazy and crazily successful
approach to diffusion for pixels of sounds plates sure thing
right so this is um going to be a little bit of Show and Tell and most of the code in the notebook is just copied and
pasted from I think Netbook 30. um but we're going to be trying to generate something other than just
images and so specifically I'm going to be loading up a data set of bird calls these are just like short samples of I
think 10 different classes of birds calling um and so we need to understand like okay well this is a totally different
domain right this is audio if you look at the data like let's look at an example of the data this is coming from
a hugging face data set so that that line of code will download it automatically if you haven't got it before right
yeah yeah so this will download it into a cache and then sort of handle um
you created this data set right did you is this already a data set you found somewhere else or you made it or what
um this is a subset that I made from a much larger data set of longer call recordings um from an open website called xenocanto
so they collect all of these sound recordings from people they have experts who help identify what birds are calling
um and so all I did was find the audio Peaks like where is there most likely to
be a bird call and clip around those just to get a smaller data set of things
where there's actually something happening yeah not a particularly amazing data set
um in terms of like the recordings have a lot of background noise and stuff but a fun small audio one to play with
um yeah and so when we talk about audio you've got a microphone somewhere it's reading like a pressure level essentially
um in the air with these sound waves and it's doing that some number of times per second so we have a sample rate and in
this case the data has a sample rate of 32 000 samples per second so every
second the waveform that's being approximated there's lots of little up across up across up across kind of
things basically yeah right yeah um and so that's great for you know
capturing the audio um but it's not so good for modeling because we now have 30 000 values per second in this one big
1D array um and so yeah you can try and find models that can work with that kind of
data but what we're going to do is a little hack and we're instead going to use something called a spectrogram
so to illustrate the original data is the main issue it's it's it's too big and slow to work with it's it's too big
um but also you have some like some sound waves are at
100 Hertz right so they're they're going up and down 100 times a second and some are at a thousand and some are ten
thousand and often there's background noise they can have extremely high frequency components and so if you're
looking just at the waveform there's lots and lots of change second to Second and there's some very long
range dependencies of like oh they're generally High here it's generally low there and so it can be quite hard to
capture those patterns um and so part of it is it's just a lot of samples to deal with
um but part of it also is that it's not like an image where you can just do like
convolution and things nearby each other tend to be related or something like that um it's quite tricky to disentangle
what's going on um and so we have this idea of something called a spectrogram
uh this is a fancy 3D visualization but it's basically just taking that audio
and mapping time on one axis so you can see as time goes by we're moving along the x-axis and then on the y-axis is
frequency and so the um the Peaks here show like intensity at different frequencies and so if I make a pure note
[Music] you can see that that screen maps in the frequency domain but when I'm talking
there's lots and lots of Peaks and that's because our voices tend to produce a lot of overtones so if I go
you can see there's a main note but there's also these subsequent notes and if I play something like a chord
[Applause] you can see this you know maybe three main Peaks and then each of those have
these harmonics as well so it captures a lot of information about the signal
um and so we're gonna turn our audio data into something like this where even
just visually if I'm a bird you can see this really nice spatial
pattern yeah and the hope is if we can generate that and then if we can find some way to turn it back into audio and
then we'll be Off to the Races and so yeah that's what I'm doing in this notebook we have
um I'm leaning on the diffusers dot
pipelines.audiodafusion.mel class and So within the realm of spectrograms there's
a few different ways you can do it so this is from the torch audio docs but this notebook is from the hugging face
diffusion models class so we had that waveform that's those raw samples and we'd like to convert that
into what they call the frequency domain um which is things like these spectrograms
um and so you can do a normal normal spectrogram a power spectrogram or something like that but we often use
something called the Mel spectrogram which is exactly the same it's actually probably what's being visualized here
and it's something that's designed to map The View like the frequency ranges into
a range that's um like tied to what human hearing is
based on and so rather than trying to capture all frequencies from you know zero Hertz to 40 000 Hertz a lot of
which we can't even hear it focuses in on the range of values that we tend to
be interested in as as humans and also it does like a transformation into into
kind of like a log space um so that the the intensities like highs and lows correspond to loud and
quiet for human hearings it's very tuned for um the types of audio information that
we actually might care about rather than you know tens of thousands of kilohertz that only bats can hear
um okay so we're going to rely on a class to abstract this away but it's going to basically give us a transformation from waveform to
spectrogram and then it's also going to help us go from spectrogram back to waveform
um and so let me show you my data I have this two image function that's going to take the audio array it's going to use
the Mel um class to handle turning that into
um spectrograms and the class also does things like it splits it up into chunks based on you can set like a desired
um resolution I'd like 128 by 128 spectrogram it says okay great I know how many I know you need 128 like
frequency bins for the frequency axis and 128 um steps on the on the time axis so it
kind of handles that converting and resizing um and then it gives us these audio
slice to image so that's taking a chunk of audio and turning it into the spectrogram and it also has the inverse
um so our data set is fairly simple we're just referencing our original
audio data set but we're calling that to image function and then returning it into a tensor and we're mapping it to
minus 1.5 to 0.5 so similarly to what we've done with like the grayscale
images in the past um so if you look at a sample from that data we now have instead of an audio
waveform of 32 000 or 64 000 if it's two seconds samples we now have this 128 by
128 pixel um spectrogram which looks like this and
it's just it's gray scale so this is just matplotlibs colors um but we can test out going from the
spectrogram back to audio using the image to audio function that the the Mel class has and that should give us
um now this isn't perfect because the spectrogram shows the intensity at different frequencies but with audio
you've also got to worry about something called the phase and so this image to audio function is actually behind the
scenes doing a kind of iterative approximation with something called the Griffin Lin
algorithm so I'm not going to try and describe that here but it's just it's approximating it's guessing what the
phase should be it's creating a spectrogram it's comparing that to the original it's updating it's doing the
sort of like iterative very similar to like an optimization thing to try and generate an audio signal that would
produce the spectrogram which we're trying to invert so just to clarify that so my understanding what you're saying
is that the spectrogram is a lossy conversion of the sound into an
image and specifically it's lossy because it tells you the kind of intensity at each
point but it's not it's kind of like is it like the difference between a sine wave and a cosine wave like they're just
shifted in different ways we don't know how much it's shifted so coming exactly the sound you do have to get that that
shifting the phase correct and so it's trying to guess something and it sounds like it's not doing a great guess from
the the original audio is also not that amazing um but yes the the spectrogram
back to audio task this these dotted lines are like highlighting this is yeah it's an approximation and there are deep
learning methods now that can do that um better or at least that sound much higher quality
um because you can train a model somehow to go from this image like representation back to a
an audio home signal and but we just use the approximation for this notebook
um okay so now that we can represent our data as like a grayscale 128 by 128
pixel image um everything else becomes very much the same as the previous diffusion models examples we're going to use this noisify
function to add different amounts of noise and so you can see now we have our spectrograms but with varying amounts of
noise added we can create a simple diffusion model I'm just copying and pasting the results
but with one extra layer um just with very few channels to go from 128 to 64 to 32 to 8 to 6 I mean to
16 by eight to eight um no attention just I think pretty much copied and
pasted from notebook 30 and train it for in this case 15 epochs it took about I
don't know this is interesting you're using simple diffusion yes
um so specifically this is the simple diffusion model that you um I think I've already introduced maybe
not um yeah I think we briefly looked at it so let's remind ourselves of what it does here
oh okay yeah um so we have some number of down blocks with a specified number of channels and then the key Insight
from simple diffusion was that you often want to concentrate the computes in the sort of middle at the low resolution
so that's these these mid blocks and they're Transformers yes
um yeah um and so we can stack some number of those and then
um the corresponding path and this is a Unix so we're passing in the features from there the down path as you go
through those up blocks um and so we're gonna go take in um
image and time step we can embed the time step we're going to go through our down blocks and saving
the results we're going to go through the mid blocks there we go through the
Note Blocks yeah before that you've also got the um embedding
of the uh locations at self.le is the loanable
embeddings using scale and shift I remember all right so this is preparing it to go
through the Transformer blocks by adding some learnable embeddings cool
um right and then we're reshaping it to be effective gear sequence since that's how
we had written our transform it to have expected 1D sequence of embeddings and
so once you've gone through those mid blocks we can reshape it back and then we go through the up blocks passing in
and also are saved outputs from the down path um yeah so it's a nice it's a nice model
you can really control how much parameters and compute you're doing just by setting like what are the number of
features or channels at each of those down block stages and how many mid blocks are you going to stack
um and so if you want to scale it up it's quite easy to term let me just add more mid blocks maybe I'll add more channels
and there's a very easy Mouse to tweak to get a larger or smaller model fun fun
thought Shadow is um simple diffusion only came out a couple of months ago and
I don't think I think ours might be the first publicly available code Forex I don't
think the authors released the code I suspect this is probably the first time maybe it's ever been used to generate
audio before possibly yeah I guess I know a couple of
people who've at least privately done their implementations when I asked the author if he was releasing Cody said uh
but it's important it's just a bunch of Transformer blocks you can get yourself I'll release it eventually
um no maybe maybe not I've done it malignant but they were like oh you can see the pseudocode it's pretty easy
apparently yeah it is pretty easy yeah cool so trains the last goes down
as we hope um sampling is exactly the same as generating images normally and that's
going to give us the spectrograms so I'm using DDM sampling with 100 steps
um and to actually listen to these samples we're then are just going to use that image to audio function again to
take our grayscale image um and in this case actually it expects a pil image so first converted it to pil
um and then turn that back into audio and so we can play some of the generated samples
wow yeah
that's so cool I don't know that I could guarantee what bird is making these calls and some of them are better than
others like some of them um yeah some of the original samples
sounds like that so exactly yeah so yeah that's generating
and fake bird calls with with um spectrogram diffusion there's projects that do this on music
um so the refusion projects
and yeah there's various other like pre-trained models that do diffusion on spectrograms to produce
um you know music clips or Voice or whatever
um I may have Frozen refusion is actually the stable diffusion model that's that's fine-tuned specifically
for for this for the spectrogram generation which is which I find very impressive it's like a model that was
originally for you know text to image is instead can also generate these spectrograms I guess there's still some
useful information in you know this sort of text image model that kind of generalizes or you can still be used for
text to audio so I found that a very interesting impressive application as well so refusion is an awesome name
yeah indeed it is yeah cool and I guess since it's a latent
model that leads us on to the next topic right I was just gonna say we've got a natural segue there yes so we're um if
we want to replicate refusion then
um we'll need latents yeah so the the final non-nlp part of
stable diffusion is this ability to use the more compressed representation
created by a vae called latents instead of pixels
so we're going to start today by creating a vae taking a look at how it works
um so to remind you as we learned back in the the first lesson of this part of
part two um the vae model converts the um
256 by 256 pixel 3 Channel into a
um is it 64 by 64 by 4 it'll be 32 if it's 256. I think it's
512 to 16. oh 512 64. okay so do a 32 by 32 by four so dramatically smaller which
makes life so much easier um which is which is really nice
um having said that you know simple diffusion does the
first you know few in fact you know all the down sampling pretty quickly and and all the hard work happens you know at a
16 by 16 anyway so maybe it's you know with simple diffusion it's not as big a deal as it used to be but it's
still you know it's very handy particularly because for us folks with more normal amounts of compute we can
take advantage of all that hard work that the uh the uh stability.ai
computers did for us by creating the the stable diffusion vae
um so that's what we're going to do today but first of all we're going to create our own um so
let's do a vae using fashion mnist so the first or the first stuff is just the
the normal one thing I am going to do for this simple example though is I'm going to flatten the
um fashion mnist pixels into a vector to make it as simple as possible
um okay so uh We've we're going to end up with vectors of length 784 because 28 by
28 784. we're going to create a single kitten
layer MLP with um 400
hidden and then 200 outputs so here's a linear layer so it's a
sequential containing a linear and then an optimal activation function then an optional normalization
we'll update init weights so that we initialize linear layers as well
so before we create a vae which is a variational autoencoder we'll create a
normal Auto encoder we've done this once before and we didn't have any luck in
fact we were so unsuccessful that we decided to go back and create a loaner and come back a few weeks later once we
knew what we were doing so here we are we're back we think we know what we're doing so we're just going to recreate an auto
encoder just like we did some lessons ago so there's going to be an encoder
which is a sequential which goes from our 768 inputs to our 400 hidden and
then a linear layer with our 400 hidden and then an output layer from the 400 hidden to the 200
outputs of the encoder so there we've got our latents
and then the decoder will go from those 200 latents to our
400 hidden have our hidden layer and then come back to our
768 inputs um
all right so we can optimize that in the usual way
using atom and we'll do it for 20 epochs runs
pretty quickly because it's quite a small data set and quite a small model um
and so what we can then do is we can grab a batch of our X who
actually grab the batch of X earlier uh way back here
so I've got my batch of images and we can put it through our model
pop it back on the CPU and we can then have a look at our
original Mini batch and we have to reshape it to 28 by 28 because we previously had flattened it
so there's our original and then we can look at the result after putting it through our model
and there it is and as you can see it's you know very roughly regenerated and so
this is um not a massive compression it's compressing it from 768 to 200.
um and it's also not doing an amazing job of recreating the original details um but you know this is the simplest
possible Auto encoder so it's doing you know it's a lot better than our previous attempt so that's good
um so what we could now do is we could just generate some noise and then we're not even going to do
diffusion we're just going to go and say like okay we've got a decoder so let's just decode that noise
and see what it creates and the answer is not
anything great I mean I could kind of recognize that might be the start of a shoe
maybe that's the start of a bag I don't know but it's not doing anything amazing so we have not successfully created an
image generator here but there's a very simple step we can do to make something that's more like an image generator the
problem is that um these 200
um this Vector of length 200 we're creating there's no particular reason that things
that are not in the data set are going to create items of clothing we haven't done
anything to try to make that happen we've only tried to make this work for the things in the data set you know and
um therefore when we just randomly generate
a bunch of you know a vector of length 200 or 16 vectors of length 300 in this
case and then decode them there's no particular reason to think that they're going to create something that's
recognizable as clothing so the way a vae tries to fix this
is by we've got the exact same encoder as before except it's just missing its
final layer its final layer has been moved over to here I'll explain why there's two of
them in a moment so we've got the inputs to Hidden the hidden to Hidden and then the hidden to latents the decoder is
identical okay latency hidden head into hidden head into inputs
um and then just as before we call the encoder
um but we do something a little bit weird next which is that we actually have two
separate final layers we've got one called mu for the final of the encoder
and one called LV which stands for log variance so our encoder has two different final
layers so we're going to call both of them Okay so we've now got two
encoded 200 long lots of latents what do we do with them
what we do is we use them to Generate random numbers
and the random numbers have a mean of mu
um so when you take a round of zero one so this creates zero one random numbers mean zero standard deviation one so if
we add mu to it they now have a mean of mu or approximately and if you multiply the
random numbers by half of log variance e to the power of
that right so given this log of variance this is going to give you standard deviation
so this is going to give you a standard deviation of e to the half LV and a mean
of mu y the half it doesn't matter too much but if you think about it
standard deviation is the square root so the variance is squared so when you take
the log you can move that um that uh half into the multiplication
because of the log trick that's why we've just got the half here instead of
the square root which would be to the power of a half um so this is just yeah this is just the
standard deviation so we've got the standard deviation times normally distributed random noise plus mu so we
end up with normally distributed
numbers we're going to have 200 of them for each element of the batch where they have a standard deviation of
the result of this final layer and a variance
which is the result or log variance so if the result of this final layer and then finally we passed that through
the decoder as usual I'll explain why we passed back three things but for now we'll just worry about the fact we passed back the result of the decoder so
what this is going to do is it's going to generate um the the result of calling encode
is going to be a little bit random on average you know it's still generating exactly
the same as before which is the result of a sequential model with you know MLP
with one hidden layer but it's also going to add some Randomness around that right so this
here's the bit which is exactly the same as before this is the same as calling in code before but then here's the bit that
adds some randomness to it and the amount of Randomness is also
itself random okay so then that gets run through the decoder
um okay so if we now just [Music] um or you know train that
right using the result of the decoder and using um I think we didn't use MSE
last we used uh binary cross entropy loss which we've seen before so if
you've forgotten you should definitely go back and re-watch that are really part one um or we've done a bit of it in
part two as well binary cross entropy loss um
with loggetts means that you don't have to worry about doing the softmax it does the soft Max for you
um so if we just um optimize this using BCE now
you would expect and it would I believe I haven't checked that it would basically take this final like this
layer here and turn these all into zeros as a result of which it would have no variance at all
and therefore it would behave exactly the same as the
previous Auto encoder does that sound reasonable to you guys yeah okay
um so that wouldn't help at all because what we actually want is we want some variants and the reason we want some
variance is we actually want to have it generate some latents which are not
exactly our data they're around our data but not exactly our data and when it generates latents that are around our
data we want them to decode to our to the same thing we want them to
decode to the correct image and so as a result if we can train that
right something that it it does include some variation and still decodes back to the original
image then we've created a much more robust model and then that's something
that we would help then that we would hope then when we say okay well now decode some noise it's going to decode
to something better than this so that's the idea of a vae so how do we get it to create
um a log variance which doesn't just go to zero
um well we have a second loss term it's called the KL Divergence loss we've got
a care called kod loss and what we're going to do is our vae loss is going to
take the branding across entropy between the
actual decoded bit so that's input 0 and the target
okay so that's this is exactly the same as before is this binary cross entropy and we're going to add it to this kld
loss K all diversions now KL Divergence the details don't matter terribly much
what's important is when we look at the kld loss it's getting past the input and
the targets but if you look it's not actually using the targets at all
so if we pull out in the the input into it three pieces which is our predicted
image our mu and our log variance we don't use this either so the BCE loss only uses the predicted
image and the actual image the KL Divergence loss only uses
mu and log variance and all it does is it returns a number which says
for each item in the batch is Mu close to zero and is log variance close to one
how does it do that well for me it's very easy mu squared so if if mu is close to zero
then minimizing mu squared does exactly that right if mu is one then mu squared
is one if mu is minus one mu squared is one If U is zero mu squared is zero
that's the lowest you can get for a squared um Okay so we've got a
mu squared piece here um and we've got a dot mean so we're
just taking that's just basically taking the mean of all the mules and then there's another piece which is we've got
log variance minus e to the power of log variance
so if we look at that so let's just grab a bunch of numbers between neg3 and 3
and do number minus e to the power of that number and I'm just going to pop in the 1 plus
and the 0.5 times as well they matter much and you can see that's got a minimum of zero
so when that's a minimum of zero e to the power of that which is what
we're going to be using it's actually half times e to the power of that but that's okay is what we're going to be using in our
um dot forward method that's going to be e to the power of 0 which is going to be
one so this is going to be minimized
where log variance x equals one so therefore this whole piece here
will be minimized when mu is zero and LV is also
zero and and so therefore LV e to the power of LV is one now the reason that
it's specifically this form is basically because
um there's a specific mathematical thing called the KL Divergence which Compares how similar to
distributions are and so a normal distribution can be fully characterized by its mean and its variance and so this
is actually more precisely calculating the similarity that specifically the KL Divergence
between the actual mu and LV that we have and a distribution with a mean of
zero and a variance of one um but you can see hopefully why
conceptually we have this mu dot power 2 and where we have this LV
dot X LV minus lv.x yeah
um so that is our Bae loss did you guys have anything to
add to any of that description so maybe to highlight the the objective
of this is to say rather than having it so that the exact point that that an input is encoded to decodes back to that
input to be saying number one the space around that point should also decode that input because we're going to
transform some variance and number two the overall variance should be like yeah the the overall
space that a user should be roughly zero mean and units and variants right so instead of
people to like map each input to like an arbitrary point and then decode only that exact point to an input we're now
mapping them to like a restricted range and we're saying that not not just each point but its surroundings as well it
should also decode back to something that looks like that image um and that's trying to like condition
this latent space to be much nicer so that any object Point within that range
will hopefully map to something useful which is a harder problem to solve right so we would expect given that this is
exactly the same architecture we would expect its ability to actually decode
would be worse than our previous attempt because it's a harder problem that we're trying to solve which is to just we've
got random numbers in there as well now but we're hoping that this ability to generate images will improve
um thanks johno okay so I actually asked Bing about this
um which is this this is more of an example of like I think for you know now that we've got gpt4 and Bing and stuff I
find they're pretty good at answering questions that like I wanted to explain to students what would happen if the
variants of the latents was very low or what if they were very high so why do we want them to be one and they thought
like oh gosh this is hard to explain so maybe Bing can help so I actually thought it's pretty good so I just see
what being said so Bing says if the variance of the latents are very well low then the encoder distribution would
be very peaked and concentrated around the means that was the thing we were describing earlier if we had trained
this without the kld loss at all right it would probably make the variance zero
and so therefore the latent space would be less diverse and expressive and limit the ability if the decoder to
reconstruct the data accurately make it harder to generate new data if it's different from the training data
which is exactly what we're trying to do and if the variance is very high then the encoder would be very spread
out and diffuse it would be more the latents would be more noisy and random
um make it easier to generate new data that's unrealistic or nonsensical okay
so that's why we want it to be exactly at a particular point
so when we train this we can just pass vae last as our loss
function but it'd be nice to see how well it's going at reconstructing the original image and how it's going at
creating a 0 1 distribution data separately so um
what I ended up doing was creating just a really simple thing called func metric which I derived from the capital M mean
um class in the torch
I'm just trying to find it here from the torch eval.metrix so they've already got something that
can just calculate means it's obviously this stuff's all very simple and we've created our own metrics class ourselves back a while ago that since we're using
torch eval I thought this is useful to see how we can create one a custom metric where you can pass in some
function to call before it calculates them in so if you call so you might remember
that the way torch eval works is it has this thing called update which gets past the input and the targets so I add to
the weighted sum the result of calling some function on the input into targets
um so we want two kind of new metrics one
is the um we're going to print it out as kld which is a func metric on kld loss someone in
which we'll print out as BCE which is a funk metricon BCE loss
um and so the actual when we call the learner the loss function we'll use is VE loss
but we're going to pass in as metrics
um this list of additional metrics to print out so I'm just going to print them out
and in some ways it's a little inefficient because it's going to calculate kld loss twice and BCE lost
twice one to print it out and one to go into the you know actual loss function but it doesn't take long for that bit so
I think that's fine so now when we call learn.fit you can see it's printing them all out so the
BCE that we got last time was 0.26
and so this time yeah it's not as good as 0.31 because it's a harder problem and it's got random randomness in it
um and you can see here that the BCE and kld are pretty similar scale when it
starts that's a good sign if they weren't you know I could always in the loss function
scale one of them up or down but they're pretty similar to start with so that's fine
so we trained this for a while and then we can use exactly the same code for sampling as before
and yeah as we suspected its ability to decode is worse
so it's actually not capturing the l e e at all in fact and the shoes got very
blurry um but the hope is that when we call it on noise called the decoder on random
noise Ah that's much better we're getting it's not amazing but we are getting some recognizable shapes so you
know vaes are you know um
not generally going to get you as good a results as um diffusion models are although actually
if you train really good ones for a really long time they can be pretty impressive but yeah even in this
extremely simple quick case we've got something that can generate recognizable
items of clothing did you guys want to add anything before we move on to the stable diffusion vae
okay so this yeah so this pie is
very crappy um and as we mentioned like one of the one of the key reasons to use a vae is
actually that you can benefit from all the compute time that somebody else has put
into training a good vae maybe just also like one one thing when
we say good vae the one that we've checked trained here is is good at generating because it Maps
down to this like one 200 dimensional vector and then back um in a very useful way and like if you
look at vaes for generating they'll often have a pretty small dimension in the middle
um and they and it'll just be like this Vector that gets mapped back up and so vae that's good for generating is
slightly different to one that's good for compressing and like the stable diffusion one we'll see has this like spatial component still it doesn't map
it down to a single Vector it Maps it down to a 64 by 64 or whatever
um I think that's smaller than the original but for generating we can't just put random noise in there and hope like a
cohesive image will come out um so it's less good as a generator but
it is good because it has this like compression and reconstruction ability cool yeah so let's take a look
um now um to demonstrate this we want to move to a
more difficult task um because we want to show off how using latent let us do stuff we couldn't do
well before um so the more difficult task we're going to do is generating bigger images
and specifically generate images of bedrooms using the L Sun bedrooms data
set so lsan is a really nice
data set which has
many many millions of images across 10 scene categories and 20 object
categories um and so very it's very rare for people to use of the object character
categories to be honest but people quite often use the the scene categories um they're a little
um well more than little okay can be extremely slow to download the the website they come from is very often down so what I did was I put a subset of
20 of them um onto AWS they kindly provide some
free data set hosting for our students um and also the original L signs in a
slightly complicated form it's in something called an lmdb database and so I turned them into just normal images in
folders so you can yeah download them directly from from the AWS data set site
that they've provided for us um so I'm just using fast core to save
it and then using Python's shutl to unpack the gzipped tar file
um okay so that's given us once that runs which is going to take
a long time um and you know if
um it might be you know even more reliable just to do this in the Shell with wget or Aria 2C or something
um then doing it through python so this will work but if it's taking a long time or whatever maybe just delete it and do
it in the Shell instead um okay so then
I thought all right how do we turn these into
latents well we could create a data set
in the usual ways it's going to have a length so we're going to grab all the files so
blob is a built into python which we'll search for in this case
star.jpg and if you've got star star slash that's going to search recursively
as long as you pass recursive so we're going to search for
um all of the JPEG files inside our data slash bedroom
folder so that's what this is going to do it's going to put them all into the files attribute and so then when we get an
item the I item it will find the I file it will read that image so this is pi
torch's read image it's um it's the fastest way to read a JPEG
image people often use Pao but it's quite hard to find a really well optimized
pil version that's really compiled fast where else the pytorch torch Vision team
have created a very very fast read image so that's why I'm using
theirs and if you pass in image read mode.rgb
it'll automatically turn any one channel black and white images into three Channel images for you or if there are
four Channel images with transparency it'll turn those so this is a nice way to make sure they're all the same and
then this turns it into floats from not to not to one and these images are generally very
close to 256 by 256 pixels so I just crop out the top 256 by 256 bit because
I didn't really care that much um and we do need them to all be the same size in order that we can then pass
them to the vae stable diffusion vae decoder as a batch otherwise it's going
to take forever so I can create a data loader that's going to go through a bunch of
them at a time um so 64 at a time and use however many CPUs I have as the
number of workers it's going to do it in parallel and so the parallel bit is the bit that's actually reading the jpegs
which is otherwise going to be pretty slow so if we grab a batch here it is here's what it looks like okay generally
speaking they're just bedrooms although we've got one pretty risque situation in the bedroom but uh on the hair on the
whole they're not not safe for work this is the first time I've actually seen a an actual bedroom scene taking place as
it were um all right so as you can see this mini
batch of um uh if I just grab the first 16 images has um three channels and 256 by 256 pixels
so that's how big that is for 16 images so that's
uh seven two eight okay so three point one
four five million floats to represent this
um okay so as we learned in the first lesson of part two we can grab [Music] um
an auto encoder directly using diffusers using from
pre-trained we can pop it onto our GPU and importantly we don't have to say
with no grad anymore if we pass the requires grad false and remember this neat trick
in pie torch if it ends in an underscore it actually changes the thing that you're calling in place so this is going
to stop it from Computing gradients which would take a lot of time and a lot of memory otherwise
so let's test it out let's encode our mini batch
and so just like jono was saying this has now made it much smaller it's got just in in our 16 batch of 16 it's now a
four Channel 32 by 32 so if we can compare the previous size
to the new size it's 48 times smaller so that's 48 times less memory it's
going to need and it's also going to be a lot less compute for a convolution to go across
that image so there's no good unless we can turn it back into the original image
so let's just have a look at what it looks like first now it's a four Channel image so we can't naturally look at it
but what I could do is just grab the first three channels and then they're not going to be between
naught and one so if I just dot sigmoid now they're between one and one and so you can see that our you know risque
bedroom scene you can still recognize it right or this bedroom this bed here you can still
recognize it so there's still that kind of like the basic geometry is still clearly there
but it's um yeah it's clearly changed it a lot as well
so importantly we can call decode on this 48 times smaller
tensor and it's really I think absolutely
remarkable how good it is I I can't tell the difference
um to the original let me if I zoom in a bit
her face is a bit blurry with her face always a bit blurry uh it's always a bit blurry
first second third oh hang on
did that used to look like a proper ND yeah okay so you can see this used to say that clearly there's an ND here
and now you can't see those letters so so and this is actually a classic thing that's known for this particular vae is
it's um not able to regenerate writing correctly at small
font sizes I think it's also pretty it's like I think we hear what the faces are already
pretty low resolution but if you are at a higher resolution the faces also would probably not uh be converted
appropriately okay cool but overall yeah it's done a great job
um a couple of other things I wanted to note was like so like you mentioned like a 40 I guess a factor of 48 degrees uh
oftentimes people will refer to mostly at the spatial resolution so since it's going from 256 to uh by 256 to 32 by 32
so that's like a factor of uh eight so they sometimes will know like I think it's like f8 or something like this
they'll note the the spatial resolution so sometimes you may see that written out like that
um and of course is an eight squared decrease in the number of pixels which is interesting right right
and then the other thing I wanted to note was that the vae is also trained
with uh with a perceptual loss objective as well as uh technically like a like a
a discriminator again objective I don't know if you were going to go into that yeah so
um yeah so perceptual loss we've we've already discussed right so the vae
is going to you know when they trained it um so I think this was trained by
confers right the you know Robin and um gang
um uh and use stability.ai donated compute
for that and uh they went so typically actually no the the VA was actually
trained separately and it's actually a trend on the open images data set and it
was just this vae that they trained by themselves on you know a small subset of data but because the VA is so powerful
it's actually able to be applied to all these other um data sets as well okay great yeah so
they um so they would have had um a a KL diversion loss and they would
have either had an MSC or BCE like something might have been an MSE loss they also had a perceptual loss which is
the thing we learned about when we talked about super resolution which is where when they compared the uh the output
images to the original images they would have run that through
a you know imagenet trained or similar classifier and confirmed that the
activations they got through that model were similar and then the final bit
um is as Tunisia was mentioning is the adversarial loss
um which is also known as a as a gan loss
so again is a generative adversarial Network and
um the the Gan loss what it does is it grabs it is actually
um more specifically what's called a patchwise patch wise
again loss and what it does is it takes like a little section of an image right and
what they've done is they train it's um let's just simplify it for a moment and
imagine that they've pre-trained a classifier right where they've basically got something that you can pass it a
real you know patch from a bedroom scene and a um and a fake patch from a bedroom
scene and they both go into
the the What's called the discriminator
and this is just a normal you know resnet or whatever which basically outputs
something that either says uh yep um uh the the image is real
or nope the image is fake so sorry I said it passes in two things you just oh that was wrong you just pass in one
thing and it returns either it's real or it's fake and specifically it's going to give you something like the probability
that it's real there is another version I don't think it's what they used to be passing to and it tells you which one's
relative do you remember tanishq is it a relativistic gan or a normal gam it's I think it's a normal one yeah so the
relativistic end is when you pass in two images and it says switch is more real the one we think that we if we remember
correctly they use as a regular can which just tells you the probability that it's real and so you can just train
that by passing in real images and fake images and having it learn to classify
which ones are real and which ones are fake um so now that once you've got that model
trained then as you train your Gan you pass in the patches you know of of each
image into the discriminator just call D here
right and it's going to spit out the probability that that's real and so
if it's bad out you know 0.1 or something then you're like oh
dear that's terrible our vae is spitting out pictures of bedrooms where the
patches of it are easily recognized as not real but the good news is that's
going to generate derivatives right and so those derivatives then is going to
tell you how to change the pixels of the original generated image to make it trick again
better and so what it'll do is it'll then use those derivatives as per usual
to update our our vae and the vae in
this in this case is going to be called a a generator right that's the thing that's generating the pixels and so the
generator gets updated to be better and better at tricking the discriminator and after a while what's going to happen
is the generator is going to get so good that the discriminator gets fooled every time right and so then at that point you
can you can fine tune the discriminator better by putting in your better
generated images right and then once your discriminator learns again how to recognize the difference between real
and fake you can then use it to find the generator and so this is kind
of ping-ponging back and forth between the discriminator and the generator back when Gans were first created
um you know people were finding them very difficult to train and actually a method we
developed at fast AI I don't know if we were the first to do it or not um was this idea of kind of pre-training
a a generator just using perceptual loss and then pre-training a discriminator to
be able to fool the generator and then ping-ponging backwards and forwards between them after that
basically whenever the discriminator got too good start using the generator anytime the generator got too good start
using the discriminator nowadays that's pretty standard I think to do it this way and
so yeah this this this this this this Gan loss which is basically saying
penalized for failing to fool the discriminator is called an adversarial
adversarial Plus
to maybe motivate why why you do this if you just did it with like a mean squared
error or even a perceptual loss um with such a high compression ratio the
vaes tend to like produce a fairly blurry output because it's not sure whether there's texture or not you know in in
this image or the edges on like super well defined where they'll be because it's going from like one four
dimensional thing up to like this whole patch of the image um and so it tends to be a little bit
blurry and hazy because it's it's kind of hitching its bits whereas that's something that the discriminator can
quite easily pick up oh like it's blurry it must be fake you know and so then it's having the discriminator that is
adversarial loss is just kind of saying like even if you're not sure exactly where this texture goes rather go with a
sharper looking texture that looks real um than with some blurry thing that's
going to maximize your MSE um and so it like tricks it into kind of Faking this high resolution
looking sharper output yeah and um
I'm not sure if we're going to come back and like train our own and at some point but um if you're you know interested in
training your own Gan or advocation called again right I mean nowadays
we never really just use again we have an adversarial loss as part of a training process so if you want to learn
how to use adversarial loss um but in detail and see the code the 2019 first
AI course lesson seven at part one um has a has a walk through so we have
sample code there and you know yeah maybe given time we'll come back to it
um Okay so quite often people will call the vae
encoder when they're training a model which to me makes no sense right because the
encoded version of an image never changes um unless you're using data augmentation
and want to do augmentation on sorry to you know encode augmented images
um I think it makes a lot more sense to just do a single run through your whole training set and encode everything once
so naturally the question is then well what do you where do you save that so it's going to be a lot of ram if you put
this leave it in Ram and also if you you know as soon as you restart your computer we've lost all that work
there's a very Nifty file format you can use called a memory mapped numpy file
which is what I'm going to use to save our latents a memory mapped numpy file
is basically what happens is you you take the memory in Ram that numpy would
normally be using and you literally like copy it
onto the hard disk basically that's what they mean by memory mapped there's a mapping between the memory and RAM and
the memory and hard disk and if you change one it changes the other and vice versa they're kind of two ways of seeing the same thing and so if you and so if
you create a memory mapped numpy array then when you modify it it's actually
modifying it on disk but thanks to the magic of your operating system
it's using all kinds of beautiful caching and stuff to not make that slower than using a normal numpy array
and it's going to be very clever at um it doesn't have to store it all in
Ram it only stores the bits in Ram that you need at the moment or that you've used recently it's really Nifty at kind
of caching and stuff so it's kind of it's like magic but it's using the your operating system to do that magic for
you so we're going to create a memory mapped file using np.memap and so it's going to be
stored somewhere on your disk right so we're just going to put it here and we're going to say okay so create a
memory map file in this place it's going to contain 32-bit floats
so write the file and the shape of this array is going to be
the size of our data set so 303 125 images and each one is four by 32 by 32.
okay so that's our memory mapped file and so now we're going to go through our
data loader One Mini batch of 24 at a time
and we're going to vae encode that minibatch
and then we're going to grab the means from its latents right we don't want
random numbers we want the actual you know the midpoints the means so this
is using the diffusers version of that pae
so pop that onto the CPU after we're done and so that's going to be mini batch of size 64.
um as Pi torch let's turn that into numpy because pytorch doesn't have a memory mapped thing as far as I'm aware but numpy does and so now that we've got
this memory mapped array called a then everything from initially from zero
up to 64 or 60 yeah 0 to 64. not including the 64. that
whole sub part of the array is going to be set to the encoded version so it's we're just it looks like we're just
changing it in memory but because this is a magic memory map file it's actually going to
save it to disk as well um so yeah that's it amazingly enough
that's all you need to create a memory mapped numpy array of our latents when
you're done you actually have to call Dot Flash and that's just something that says like anything that's just in Cache
at the moment make sure it's actually written to disk and then I delete it because I just want
to make sure that then I read it back correctly so that's only going to happen once if the path doesn't exist
and then after that this whole thing will be skipped and instead we're going to call mp.memap again
with their own path but this time in the same data type and the same shape this time we're going to read it mode equals
R means read it and so let's check it let's just grab
the first 16 latents that we read and decode them
and there they are okay so this is like not a very
well known technique I would say sadly but it's a really good one
um you might be wondering like well what about like compression like shouldn't you be zipping them or something like
that but actually remember these latents are already the whole
point is they're highly compressed so generally speaking zipping latents from a good VA he doesn't do much
um because they're like they kind of almost look a bit random number-ish
Okay so we've now saved our entire lsan bedroom it's a 20 subset the bit that
I've provided now latents so we can now run it through
and this is the nice thing we can use exactly the same process from here on in as usual Okay so we've got the noiseify
of our usual collated version um now the blatants are
much higher than one standard deviation so if we about divided by five that takes it back to a standard deviation of
about one um I think in the paper they use like 0.18 or something but this is close
enough to make it a unit standard deviation um
so we can split it into a training and a validation set so just grab the first 90
of the training set in the last 10 percent for the validation set so those are our data sets we use a batch size of
128 so now we can use our data loaders class we created with the get DLS we created
so these are all things we've created ourselves with the training set the validation set the batch size
and alkylation function so yeah it's kind of nice it's amazing you
know how easy it is like you know a data set has the same interface as an Umpire
array or a list or whatever so we can literally just use the numpy array directly as a data set which I think is
really neat this is why it's useful to know about these like foundational Concepts because you don't have to start
thinking like oh I wonder if there's some torch Vision thing to use mem map nump profiles or something it's like oh
wait they already do provide a data set interface they don't have to do anything I just use them
so that's pretty magical so we can test that now by grabbing a
batch and so this is being noiseified and so here we can see our noiseified images and so here's
something crazy is that we can actually decode noiseified images
and so you know here's I guess this one wasn't noisified much because it's a recognizable bedroom
and this is what happens when you just decode random noise something in between
so I think that's pretty fun um yeah this next bit is always copied
from our previous notebook create a model initialize it train for a while so this took me a few
hours on a single GPU everything I'm doing is on a single GPU literally nothing in this course other than the
stable diffusion stuff itself is trained on more than one GPU um the loss is much higher than usual
and that's not surprising because it's trying to generate latent pixels which
rare like it's it's it's much more precise as to exactly what it wants you know it's not like
lots of pixels where the ones next door are really next to each other really similar or the whole background looks
the same or whatever a lot of that stuff it's been compressed out it's a more difficult thing to predict latent pixels
so now we can sample from it in exactly the same way that we always have using ddam
but now we need to make sure that we decode it because the thing that it's
predict that it's sampled latents because the thing that we asked it to learn to predict a latents
and so now we can take a look and we have
bedrooms ah and some of them look pretty good I
think this one looks pretty good I think this one looks pretty good this but I don't have any idea what it
is um and this one like clearly there's bedroomy bits but
there's something I don't know there's weird bits um so the fact that we're able to create
256 by 256 pixel images where at least some of them look
quite good in a couple of hours I can't remember how long it took to train but it's a small number of hours and a single GPU
is something that was not previously possible and we're in a sense we're totally
cheating because we're using the stable diffusion vae to do a lot of the hard
work for us um but that's fine you know because that vae knows how to create all kinds of
natural images and drawings and portraits and oil paintings or whatever so
you can I think work in that latent space quite comfortably
yeah do you guys have anything you wanted to
add about that oh actually tanishq I know you've trained this for longer I only trained it for 25 epochs how long
did you how many hours did you train it for because you did you did 100 epochs right yes I did 100 it blocks I didn't
keep trying exactly but I think it was about 15 hours on an a100 okay a single a100 yeah uh
I'd argue I mean the results yeah I'll show it it's it's um it's
it's yeah it's I guess maybe slightly better um but you know I guess you can
the good ones are certainly slightly better yeah yeah like the bottom left one
better than any of mine I think so it's possible maybe at this point we just need to use more data I guess
because I guess we were using a 20 subset so maybe having more of that data to provide more diversity or something
like that maybe yeah or maybe have you tried doing the diffusers one for 100
yeah so I've got um all right so I'll share my screen if you want to stop sharing yours
um so I do have
if we get around to this maybe we can add the results back to this notebook because I do have a version that uses diffusers so everything else is
identical um 25 epochs except for the model
for the previous one I was using our our own Unit Model so we have to change
the channels now to four and a number of filters I think I might have increased it a bit
um so then I tried using yeah the diffuses unit just with whatever their defaults were
and so I got what did I get here 243 with diffusers I got
a little bit better 239 um
and um yeah I don't know if they're obviously better or not like this is a bit weird
this is a bit weird um I think like
um actually another thing we could try maybe is um do 100 epochs but use the diffusers
number of channels and stuff that they used for because I think the defaults that they use actually for diffusers is
not the same as stable diffusion so maybe we could try stable diffusion matched unit for 100 epochs and if we
get any nice results maybe we can paste them into the bottom to show people yeah
yeah cool um yeah do you guys have anything else
to add at this point all right
so I'll just mention one more thought in terms of like a bit of a interesting
project people could play with [Music] um I don't know if this is too crazy I
don't think it's been done before but my thought was like there was a huge difference in our super
resolution remember a huge difference in our super resolution results when we used a pre-trained model
and when we used um perceptual loss but particularly when we used a pre-trained model
um I thought we could use a pre-trained model but we would need a pre-trained
latents model right we would want something where our you know down sampling backbone
was pre-trained model on latents and so I just wanted to show you what
I've done and you guys you know if anybody watching wanted to try taking this further I've just done the first
bit for you to give you a sense which is I pre-trained an imagenet model not tiny imagenet but a full image net model on
latents as a classifier and if you use this as a backbone you know and also try
maybe some of the other tricks that we found helpful like having res Nets on the cross connections There's real
things that I don't think anybody's done before I don't know the scientific literature is vast and I might have
missed it but I've not come across anybody do these tricks before
um so obviously like we're one of the interesting parts of this which is designed to be challenging is
that we're using bigger data sets now but they're data sets that you can absolutely like run on a single GPU you
know um a few tens of gigabytes which fits on
any modern hard drive easily so these like uh good tests of your ability to
kind of like move things around and if you're somewhere that doesn't have access to a decent internet connection
or whatever this might be out of the question in which case don't worry about it but if you can yes try this because
it's good practice I think to make sure you can use these larger data sets
um so imagenet itself you can actually grab from kaggle nowadays
um so they call it the object localization challenge but actually this contains the Full image net data set
well the the version that's used for the imagenet competition um so I think you know people do
anything in that one case you just have to accept the terms because it has like some distribution
terms yeah exactly so you've got a kind of sign in and then join the competition and then yeah accept the terms
um so you can then download the data set or you can also download it from hugging
face [Music] um it'll be in a somewhat different format
but that'll work as well so I think I grabbed my version from from kaggle so
on kaggle you know it's just a zip file you unzip it and it creates an IL svrc
um directory which I think is what they called the competition yeah image net large scale visual recognition challenge
okay um so then um inside there there is a data and
inside there there is a CLS lock and that's actually where the that's where actually everything is going to be
um so just like before I wanted to turn these all into latents so I created in
that directory I created a latency subdirectory and this time partly just to demonstrate how these things
work I wanted to do a slightly different way okay so again we're going to create our
pre-trained VRE plop it on the GPU turn off gradients for it and I'm going to
create a data set now one thing that's a bit weird about this is that because this is really
quite a big um uh data set like it's got 1.3 million
files the thing where we go glob star star.jpg takes a few seconds you
know and particularly if you're doing this on like um you know an AWS file system or something
it can take really quite a long time um on mine it only took like three seconds but I don't want to wait three seconds so I you know a common trick for
these kind of big things is to create a cache which is literally just a list of the files so that's what this um this is
so I decided that Z pickle means a g zipped pickle so what I do is if if the
cache exists we just gzip.open the files
if it doesn't we use blob exactly like before to find all the files and then we also
save a gzip file containing pickle.dump files so pickle.dump is what we use in
Python to take basically any python object list of dictionaries and
dictionary of lists whatever you like and save them and it's super fast right and I use gzip with compressed level one
to basically be like compress it pretty well but pretty fast so
this is a really nice way to create a little cache of that
um so this is the same as always and so our get item is going to grab the file
it's going to read it in turn it into a float and what I did here was you know
I'm doing a little bit lazy but I just decided to Center crop the middle you know um so let's say it
was a 300 by 400 file it's going to Center crop the middle 300 by 300 section and then resize it to 256 by
256. um so they'll be the same size
so yeah we can now oh I managed to create the vau twice let's delete that
one so I can now just confirm I can grab a batch from that data loader encode it
and here it is and then decode it again and here it is surpass the first category must have been computer or
something so here's as you can see the ve is doing a good job of decoding pictures of computers
so I can do something really very similar to what we did before if we haven't got that destination directory
yet created go through our daily loader encoder batch and this time I'm not using a memapped
file I'm actually going to save separate um numpo files for each one so go
through each element of the batch each item um so I'm going to save it into the
destination directory which is the latents directory and I'm going to give it exactly the same path as the original one contained because it
contains the you know um uh the folder of like what what the
label is um uh make sure that the directory exists that we're saving it to and Save
that just is a numpy file this is another way to do it so this is going to be a separate numpy file for each item
does that make sense so far okay cool
so I could create a thing called a numpy data set which is exactly the same as our images data set
but to get an item we don't have to use you know open a JPEG anymore we just
call mp.load so this is a nice way to like take something you've already got and change it slightly
so it's going to return the let's do this file Jeremy just out of Interest
sorry whether you do this versus the memory mapped file was it just a different way
just to show a different way yeah yeah absolutely no particularly good reason
honestly um yeah I like to kind of like demonstrate different approaches and I
think it's good for people's python coding if you make sure you understand what all the lines of code do and yeah they both they
both work fine actually um it's partly always also forward to my own experimental interest it's like oh
which one seems to kind of feel better um
yeah all right so create out training and validation data sets by grabbing all
the numpy files and so the training and validation folders and um
then I'm going to just create a training data loader for the training data set just to see what the mean and standard
deviation is on the channel Dimension so this is every Dimension except Channel what I mean over and so there it is and
as you can see there the mean and standard deviation are not close to zero and one so we're going to store away that mean
and standard deviation such that we then we've seen transform
data set before this is just applying a transform to a data set we're going to apply the normalization transform in the past
we've just we've used our own normalization that torch Vision has has one as well so this is just
demonstrating how to just use torch Visions version but it's literally just subtracting the mean and dividing by the
standard deviation um we're also going to apply some data
augmentation we're going to use the same trick we've used before for images that are very
small which is we're going to add a little bit of padding and then randomly crop our original image size from that
so it's just like shifting it slightly each time and we're also going to use our random
erasing and it's nice because we did it all with broadcasting this is going to apply equally well to a four Channel
image as it is to a three or I think we did originally for one um
now you know I don't think anybody as far as I know has built classifiers from latents before so like I didn't even
know if this is going to work so I visualized it so so we created from X
and a different y so for two from X you can optionally add augmentation and if
you do then apply the augmentation transforms now this is going to be applied one image at a time but
augmentation transforms some of them expect a batch so we create a extra unit
axis on the front to be a batch of one and then remove it again and then Tiff and Y very much like we've
seen before we're going to turn those path names into IDs
so there's our validation and training transform data sets so so that we can look at our results we
need a denormalization um so let's create our data loaders
and grab mini batches and show us and so I was very pleased to
see that the random arrays Works actually extremely nicely so you can see you get these kind of like weird
patches you know weird patches um but it's still they're still
recognizable so this is this is like something I very very often do is to and select oh is this like thing I'm doing
in computer vision reasonable it's like what can my human brain recognize it so if I couldn't recognize this with a
drilling platform myself that I shouldn't expect a computer to be able to do it or that this is a compass
or whatever um I'm so glad they got otters so cute and you can see the cropping it's done
has also been fine like it's a little bit of a fuzzy Edge but basically like it's not destroying
the image at all they're still recognizable it's also a good example here of how
difficult like this problem is like the fact that this is Seashore I would have called this Surfers you know but maybe
surface is not an imagenet category but um yeah
okay this could be food but actually it's a refrigerator
okay so our augmentation seems to be working well so then I um yeah basically I've just copied and
pasted you know our basic pieces here and I've kind of wanted to have it all in one place just to remind myself of exactly what it is
so this is the pre-activation version of convolutions the reason for that is if I want this to be a backbone for a
diffusion model or a unit then I remember that we found that
pre-activation works best for units so therefore our backbone needs to be trained with pre-activation so we've got
pre-activation conf got a res block I rest blocks
model with dropouts this is what is copied from previous
um so I decided like I wanted to try to you know use the basic trick that we
learned about from simple diffusion of trying to put most of our work in the
later layers so the first layer just says one block then two blocks
and then four blocks and then I figured that we might then delete these final
blocks these maybe you're going to just end up being for classification this might end up being our pre-trained backbone or maybe we keep them I don't
know you know it's like as I said this hasn't been done before so anyway I try to design it in a way
that we've got some you know we can mess around a little bit with how many of these we we keep and so
also I try to use very few channels in the first blocks and so I jump up for the channels that
are aware of the work's going to do I jump from 128 to 512.
um so that's why I designed it this way you know I haven't even taken it any further than this so I don't know if it's going to be a useful backbone or
not I didn't even know if this is going to be possible to classify it seemed very likely it was possible to classify
even based on the fact that you can still kind of recognize it almost like I could probably recognize it's a
computer Maybe so I thought it was going to be possible but yeah this is all new
so that was the model I created and then I yeah trained it for 40 epochs
um and you can see after one epocket was already 25 accurate
and that's it recognizing which one over a thousand categories is it so I thought that was pretty amazing and so after 40
epochs I ended up at 66 which is really quite
fantastic because a resnet 34 is kind of like 73 or 74 accuracy when
trained for quite a lot longer um you know so to me this is extremely
encouraging that you know this is a really pretty good resnet at recognizing images
from their latent representations without any decoding or whatever so from here
you know if you want to you guys could try yeah building a better
bedroom diffusion model or whatever you like let's not be bedrooms actually
um one of our colleagues Molly um I'm just going to find it
so one of our colleagues Molly actually used the um do you guys remember was it the Celeb
faces that she used
so there's a celeb a HQ data set that consists of
images of faces of celebrities and so what Molly did was she basically yeah
used this exact notebook but used this faces data set instead and
although this one's really pretty good isn't it you know um this one's really pretty good they
certainly look like celebrities that's for sure um so yeah you could try this data set
um or whatever but um yeah try yeah maybe try it with the pre-trained backbone try it with um resonates on the
cross connections with all the tricks we used in super res try it with perceptual loss some folks we spoke to about the
perceptual I think it it won't help with latents because the underlying vae
was already trained with perceptual loss but we should try you know or you guys to try all these things
um yeah so be sure to check out the Forum as well to see what else the people who've already tried here because
it's a whole new world but it's just an example of the kind of like fun research ideas I guess we can play with
um yeah what do you guys think about this are you like surprised that we're able to quickly get this kind of accuracy for enlightens or
do you think this is a useful research path what are your thoughts yeah I think it's very interesting go
ahead the Legends are already like a a slightly compressed richer representation of an image right
um so it makes sense that that's a useful thing to train on um 66 I think Alex net is like 63 or
something like that so you know we're already at state of the art but eight years ago whatever
um it might be more like 10 years ago I know Titan passes quickly
yeah yeah I guess next year yeah next year it is 10 years ago um but yeah I'm kind of curious with the
pre-training the whole the whole value for me for like using a pre-trained network was someone else has done lots
and lots of compute on image to learn some features and now I'm gonna use that so it's kind of funny to be like oh well
let's pre-train for ourselves and then try and use that um I'm curious whether
like how best you'd allocate that compute whether you should if you've got 10 hours of GPU just do 10 hours of
training versus like five hours of pre-training and five hours of training um I mean based on our super is
thing the pre-training like it was so much better so that's why I'm feeling
somewhat hopeful about this direction yeah yeah I'm really curious to see how it goes
I guess I was gonna say it's like yeah I think there's just a lot of opportunities for I guess the late 10
doing stuff in the latents um and like I guess maybe like yeah you could I mean
here are your trade classifier as a backbone but you could think of like credit classifiers on other things for
you know guidance or things like this yeah of course we've done some experiments with that I know jono has
his Midview guidance approach for some of this sort of things but uh there are different approaches that you can play
around here um that you know exploring in the latent space can can make it computationally
cheaper than you know having to decode it anytime you want to you know like you have to look at the image and then maybe
apply a classifier or apply some sort of guidance on the image but if you can do it directly in the latent space a lot of
interesting opportunities there as well yeah and you know now we're showing that indeed yes
everything on latents models like that's something I've done to make a latent clip is just have it
like try and mirror an image space clip and so for classifiers as well you could like distill an imagenet classifier
rather than just having the label you try and like copy the logins um and then that's like an even richer
signal like you get more value per per example um so then you can create your you know
your blatant version of some existing image classifier or object detector or yeah like multimodal model like clip
um I feel like I feel funny about this because I'm like both excited about simple diffusion on the basis that it
gets rid of latents but I'm also excited about latency on the basis it gets rid of most of the pixels
I don't know how I can be cheering for both but somehow I am
I guess May the best method win so um you know for folks that are finishing
this course well first of all congratulations because it's been a journey
particularly part two it's a journey which requires a lot of patience and tenacity
um you know if you've kind of zipped through by binging on the videos that's totally fine it's a good approach but
you know maybe go back now and do it more slowly and do the you know build build it yourself and really
experiment um but assuming you know for folks who have got to the end of this and feel
like okay I I get it more or less yeah do you guys have any sense of like
what kind of things make sense to do now you know where
would you guys go from here I think that uh great opportunities
implementing papers that I guess come along these days and and and I think at
this point my way yeah
um but also at this stage I think you know we're already discussing research ideas and it I think you know we're in a
solid position to come up with our own research ideas and explore explore those ideas so I think that's a that's a real
opportunity that we we have here yeah but yeah I think that's best done often collaboratively so I'll just you know
mention that um fast AI has a Discord which if you've got to this point then
you're probably somebody who would benefit from from being there um and yeah just pop your head in and
say like there's an introduction straight to say hello and you don't you know maybe say what you're interested in
or whatever because it's it's nice to work with others I think um I mean both jono and tanishq I only
know because of the the Discord and the forums and so forth so that would be one and we also have a we
have a generative channel so anything related to generative models that that's the place so for example Bali was
posting some of her experiments in that channel uh I think there are other fast AI members posting their experiments so
if you're doing anything generative model uh related that's a great way to also get uh feedback and thoughts from
from the community yeah I'd also say that like this this if if
you're at the stage where you finish this course you actually understand how diffusion models work you've got a good
handle on what the different components and like stable diffusion are and you know how to Wrangle data for training
and play for these things you're like so far ahead of most people who are building in the space and I've got lots
of lots of companies and people reaching out to me to say do you know anybody who has like more than just oh I know how to
like load stable diffusion and make an image like you know someone who knows how to actually like Tinker with it and make it better and if you've got those
skills like don't feel like oh I'm definitely not qualified to like apply or like there's lots of stuff where yeah
just taking these ideas now and like just simple sensible ideas that we've covered in the course that have come up
and saying like oh actually maybe I could try that maybe I could play with this you know take this experimentalist
approach I feel like there's actually a lot of people who would love to have you helping them build
the million and one little stable diffusion based apps or whatever right everyone's working out and particularly
like the thing we always talk about at first AI which is particularly if you can combine that with your domain
expertise you know whether it be from your your hobbies or your work in some
completely different field or whatever you know there'll be lots of interesting
ways to combine you know you probably are one of the only people in the world right now that understand your areas of
of passion or of vocation as well as
these techniques so um and again that's a good place to kind of get on the Forum or the Discord or
whatever and start having those conversations because it can be yeah it can be difficult when you're at The Cutting Edge which you
now are by definition all right well
we better go away and start um figuring out how on Earth gpt4 Works I
don't think we're going to necessarily build the whole dpt4 from scratch uh at least not at that scale but
I'm sure we're going to have some interesting things happening with NLP and um jono Denise thank you so much it's
been a real pleasure it was nice doing things with the um with a live audience but I gotta say I
really enjoyed this uh experience of doing stuff with you guys the last two lessons so thank you so much
yeah thanks for having us it was really really fun all right bye
see you in part three bye
