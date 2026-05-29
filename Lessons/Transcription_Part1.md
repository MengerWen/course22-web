# Practical Deep Learning for Coders: Lesson 1 - YouTube
https://www.youtube.com/watch?v=8SF_h3xF3cE

Transcript:
Welcome to Practical Deep Learning for coders,  lesson one. This is version five of this course,   and it's the first new one we've done  in two years. So, we've got a lot of   cool things to cover! It's amazing how much has  changed. Here is an xkcd from the end of 2015.   Who here has seen xkcd comics before?  …Pretty much everybody. Not surprising.  
So the basic joke here is… I'll let you  read it, and then I'll come back to it. So, it can be hard to tell what's easy and what's  nearly impossible, and in 2015 or at the end of   2015 the idea of checking whether something  is a photo of a bird was considered nearly   impossible. So impossible, it was the basic  idea of a joke.
Because everybody knows that   that's nearly impossible. We're now going to build  exactly that system for free in about two minutes! So, let's build an “is it a bird” system. So, we're going to use python, and I'm  going to run through this really quickly.   You're not expected to run through it with me  because we're going to come back to it.
But   let's go ahead and run that cell. Okay, so what we're doing is we're  searching DuckDuckgo for images of   bird photos and we're just going to grab one and so here is the url of  the bird that we grabbed. Okay, we'll download it. Okay, so there it is.
So we've grabbed a bird  and so okay we've now got something that can   download pictures of birds. Now we're  going to need to build a system that can   recognize things that are birds versus  things that aren't birds, from photos.   Now of course computers need numbers to work  with, but luckily images are made of numbers. I actually found this really  nice website called pixby   where I can grab a bird, and if I wiggle over it  (let's pick its beak) you'll see here that that   part of the beak was 251 brightness of red, 48  of green, and 21 of blue. So that's RGB. And so  
you can see as I wave around, those colors are  changing (those numbers). And so this picture,   the thing that we recognize as a picture, is  actually 256 x 171 x 3 numbers, between 0 and 255,   representing the amount of red, green and blue  on each pixel. So that's going to be an input   to our program, that's going to try and figure  out whether this is a picture of a bird or not.
Okay, so let's go ahead and run this cell, which  is going to go through… (and I needed bird and   non-bird but you can't really search Google  images or DuckDuckGo images for not a bird,   it just doesn't work that way. So I just decided  to use forest - I thought okay pictures of forest   versus pictures of birds sounds like a good  starting point.
) So I go through each of:   forest, and bird. And I search for forest  photo and bird photo, download images,   and then resize them to be no bigger than 400  pixels on a side - just because we don't need   particularly big ones and it takes a surprisingly  large amount of time just for a computer to open   an image. Okay, so we've now got 200 of each.
  I find when I download images I often get a few   broken ones and if you try and train a model  with broken images it will not work. So here's   something which just verifies each image and  unlinks - so deletes the ones that don't work Okay, so now we can create what's called  a data block. So after I run this cell   you'll see that I've basically… I'll go through  the details of this later, but… a data block   gives fast.ai (the library) all the information  it needs to create a computer vision model.
And   so in this case we're basically telling it… get  all the image files that we just downloaded.   And then we say show me a few  up to six, and let's see… yeah,   so we've got some birds, forest, bird,  bird, forest. Okay, so one of the nice   things about doing computer vision models is  it's really easy to check your data because   you can just look at it - which is not  the case for a lot of kinds of models.
Okay, so we've now downloaded 200 pictures  of birds, 200 pictures of forests,   so we'll now press run. And this model  is actually running on my laptop,   so this is not using a vast data center.  It's running on my presentation laptop. And   it's doing it at the same time as my laptop is  streaming video, which is possibly a bad idea.
And so what it's going to do is it's going  to run through every photo out of those 400,   and for the ones that are forest it's going  to learn a bit more about what forest looks   like and for the ones that are bird it'll  learn a bit more about what bird looks like.   So overall it took under 30 seconds,  and believe it or not, that's enough to   finish doing the thing which was  in that xkcd comic.
Let's check   by passing in that bird that we downloaded at the  start. This is a bird. Probability it's a bird:   1.0000 (rounded to the nearest four decimal  places). So something pretty extraordinary   has happened since late 2015, which is literally  something that has gone from so impossible it's   a joke to so easy that I can run it on my laptop  computer in (I don't know how long it was) about   two minutes.
And so hopefully that gives you  a sense that creating really interesting,   you know real working programs with deep learning  is something that… doesn't take a lot of code,   didn't take any math, didn't take more than my  laptop computer. It's pretty accessible in fact.   So that's really what we're going to be  learning about over the next seven weeks. So where have we got to now  with deep learning? Well   it moves so fast, but even in the last few weeks  we've taken it up another notch as a community.  
You might have seen that something called DALLꞏEꞏ2  has been released which uses deep learning to   generate new pictures. And I thought this was an  amazing thing that this guy nick did where he took   his friends twitter bios and typed them into the  DALLꞏEꞏ2 input and it generated these pictures.
So   this guy's… he typed in commitment, sympathetic,  psychedelic, philosophical, and it generated   these pictures. So I'll just show you  a few of these. I'll let you read them… I love that. That one's pretty amazing I reckon! actually. I love this. Happy Sisyphus has actually  got a happy rock to move around.
So this is like, um, yeah, I don't know. When  I look at these I still get pretty blown away   that this is a computer algorithm using nothing  but this text input to generate these arbitrary   pictures. In this case of fairly, you know,  complex and creative things. So the guy who   made those points out, this is like… he spends  about two minutes or so, you know, creating   each of these.
Like he tries a few different  prompts and he tries a few different pictures,   you know, and so he's given an example here of…  like when he types something into the system…   like, here's an example of like 10 different  things he gets back when he puts in “expressive   painting of a man shining rays of justice and  transparency, on a blue bird twitter logo.” So it's not just, you know, DALLꞏEꞏ2, to be  clear.
There's, you know, a lot of different   systems doing something like this now. There's  something called MidJourney, which this twitter   account posted: a “female scientist with a  laptop writing code, in a symbolic, meaningful,   and vibrant style.” This one here is “an HD  photo of a rare psychedelic pink elephant.”   And this one I think is the second one here  (I never know how to actually pronounce this.
)   This one's pretty cool “a blind bat with big  sunglasses holding a walking stick in his hand.” And so when actual artists, you know, this for  example, this guy said he knows nothing about art,   you know he's got no artistic talent, it's just  something you know, he threw together. This guy is   an artist who actually writes his own software  based on deep learning and spends, you know,   months on building stuff, and as you can see,  you can really take it to the next level.  
It's been really great actually to see how  a lot of fast.ai alumni with backgrounds as   artists have gone on to bring deep learning and  art together, and it's a very exciting direction.   And it's not just images to be clear, you know one  of another interesting thing that's popped up in   the last couple of weeks is google's pathways  language model which can take any arbitrary   english as text, a question, and can create  an answer which not only answers the question   but also explains its thinking (whatever it  means for a language model to be thinking.)  
One of the ones I found pretty amazing was that  it can explain a joke. I'll let you read this… So, this is actually a joke that probably needs  explanations for anybody who's not familiar with   TPUs. So it, this model, just took the text  as input and created this text as output.   And so you can see, you know, again, deep learning  models are doing things which I think very few,   if any of us, would have believed would be maybe  possible to do by computers even in our lifetime This means that there is a lot of  practical and ethical considerations.  
We will touch on them during this course  but can't possibly hope to do them justice.   So I would certainly encourage you to check  out ethics.fast.ai to see our whole data ethics   course, taught by my co-founder Dr Rachel Thomas,  which goes into these issues in a lot more detail. All right, so as well as being an AI researcher  at the University of Queensland and fast.
ai,   I am also a homeschooling primary school teacher  and for that reason I study education a lot.   One of the people who I love in education is a  guy named Dylan Williams and he has this great   approach in his classrooms of figuring  out how his students are getting along,   which is to put a coloured cup on their desk  - green to mean that they're doing fine,   yellow cup to mean I'm not quite sure, and a red  cup to mean I have no idea what's going on.
Now   since most of you are watching this remotely  I can't look at your cups and I don't think   anybody bought coloured cups with them today,  so instead we have an online version of this.   So what I want you to do is go to  cups.fast.ai/fast - that's cups.fast.ai/fast -   and don't do this if you're, like, a fast.
ai  expert who's done the course five times - because   if you're following along that doesn't really mean  much obviously. This is really for people who are,   you know, not already fast.ai experts.  And so click one of these colored buttons. And what I will do, is I will go to the teacher  version and see what buttons you're pressing.   All right! So, so far people are feeling we're  not going too fast on the whole.
We've got one,   nope not one, brief red. Okay! So, hey Nick,  this url, the same thing with teacher on the end,   if you can you keep that open as well and let  me know if it suddenly gets covered in red.   If you are somebody who's red, I'm not going  to come to you now because there's not enough   of you to stop the class.
So it's up to you to  ask on the forum or on the youtube live chat,   and there's a lot of folks luckily  who will be able to help you, I hope. All right! I wanted to do a big shout out  to Radek. Radeck created cups.fast.ai for   me. I said to him last week I need a way  of seeing coloured cups on the internet   and he wrote it in one evening.
And  I also wanted to shout out that Radek   just announced today that he got a job at  Nvidia AI and I wanted to say, you know,   that fast.ai alumni around the world very  very frequently, like every day or two,   email me to say that they've got their dream job.  Yeah… If you're looking for inspiration on how to   get into the field I couldn't recommend nothing…  nothing would be better than checking out Radek's   work. And he's actually written a book about his  journey.
It's got a lot of tips in particular   about how to take advantage of fast.ai -  make the most of these lessons. And so I   would certainly… so check that out as well. And  if you're here live he's one of our TAs as well   so you can say hello to him afterwards.  He looks exactly like this picture here. So I mentioned I spent a lot of time studying  education both for my home schooling duties   and also for my courses, and you'll see  that there's something a bit different,   very different, about this course… which is that  we started by training a model. We didn't start  
by doing an in-depth review of linear algebra  and calculus. That's because two of my favorite   writers and researchers on education Paul Lockhart  and David Perkins, and many others talk about how   much better people learn when they learn with  a context in place. So the way we learn math   at school where we do counting and then adding  and then fractions and then decimals and then   blah blah blah and you know, 15 years later  we start doing the really interesting stuff   at grad school. That is not the way most people  learn effectively. The way most people learn  
effectively is from the way we teach sports, for  example, where we show you a whole game of sports.   We show you how much fun it is. You go and  start playing sports, simple versions of them,   you're not very good right and then you  gradually put more and more pieces together.   So that's how we do deep learning.
You will go  into as much depth as the most sophisticated,   technically detailed classes you'll find - later,  right! But first you'll learn to be very very good   at actually building and deploying models. And you  will learn why and how things work as you need, to   get to the next level.
For those of you that have  spent a lot of time in technical education (like   if you've done a phd or something) will find this  deeply uncomfortable because you'll be wanting to   understand why everything works from the start.  Just do your best to go along with it. Those of   you who haven't will find this very natural.  Oh! And this is Dylan Wiliam, who I mentioned   before - the guy who came up with the really cool  cups things.
There'll be a lot of tricks that have   come out of the educational research literature  scattered through this course. On the whole I   won't call them out, they'll just be there, but  maybe from time to time we'll talk about them. All right! So before we start talking about how  we actually built that model and how it works,   I guess I should convince you that I'm worth  listening to.
I'll try to do that reasonably   quickly, because I don't like tooting my own  horn, but I know it's important. So the first   thing I mentioned about me, is that me and my  friend Silvain wrote this extremely popular book   “Deep Learning for Coders” and that book is what  this course is quite heavily based on. We're not   going to be using any material from the book  directly, and you might be surprised by that,   but the reason actually is that the educational  research literature shows that people learn things   best when they hear the same thing in multiple  different ways. So I want you to read the book  
and you'll also see the same information presented  in a different way, in these videos. So on,e of   the bits of homework after each lesson will  be to read a chapter of the book. A lot of   people like the book. Peter Norvig, Director of  Research, loves the book. In fact his ones here   “one of the best sources for a programmer to  become proficient in deep learning.
” Eric Topple   loves the book. Hal Varian Emeritus Professor at  Berkeley, Chief Economist, Google, likes the book.   Jerome Pecente who is the head of AI at Facebook  likes the book. A lot of people like the book, so   hopefully you'll find that you like this material  as well.
I've spent about 30 years of my life   working in and around machine learning including  building a number of companies that relied on it.   And became the highest ranked competitor in the  world on Kaggle in machine learning competitions. My company Enlitic, which I founded, was the  first company to specialize in deep learning for   medicine, and MIT voted it one of the 50 smartest  companies in 2016, just above Facebook and Spacex.  
I started fast.ai with Rachel Thomas and that  was quite a few years ago now, but it's had   a big impact on the world already, including work  we've done with our students, has been globally   recognized, such as our win in the DAWNBench  competition which showed how we could train big   neural networks faster than anybody in the  world, and cheaper than anybody in the world.  
And so that was a really big step in 2018,  which actually made a big difference. Google   started using our special approaches in  their models. Nvidia started optimizing   their stuff using our approaches. So  it made quite a big difference there.   I'm the inventor of the ULMFiT algorithm  which according to the Transformers book   was one of the two key foundations behind  the modern NLP revolution.
This is the   paper here. And actually, you know, interesting  point about that, it was actually invented for   a fast.ai course. So the first time  it appeared was not actually in   the journal. It was actually in lesson four   of the course, I think, the 2016  course, if I remember correctly. And, you know, most importantly of course, I've  been teaching this course since Version One.
And   this is actually, I think, this is the very  first version of it (which even back then   was getting hbr's attention) a lot of people have  been watching the course, and it's been, you know,   really widely used. Youtube doesn't show likes  anymore, so I have to show you our likes for you.   You know it's been amazing to see how many  alumni have gone from this to, you know,   to really doing amazing things, you know.
  And so for example Andrej Karpathy told me   that at Tesla, I think he said, pretty  much everybody who joins Tesla in AI   is meant to do this course. I  believe at OpenAI, they told me that   all the residents joining there first do  this course. So this, you know this course,   is really widely used in industry and research  for people, and they have a lot of success.  
Okay, so there's a bit of brief information about  why you should hopefully keep going with this. All right so let's get back to what's  happened here. Why are we able to create   a bird recognizer in a minute or two?  And why couldn't we do it before?   So I'm going to go back to 2012 and in 2012  this was how image recognition was done.  
This is the computational pathologist - it was  a project done at Stanford. A very successful,   very famous project that was looking at the  five-year survival of breast cancer patients   by looking at their histopathology image slides.  Now, so this is, like, what I would call a classic   machine learning approach.
And I spoke to  the senior author of this, Daphne Koller,   and I asked her why they didn't use deep  learning and she said “well it just,   you know, it wasn't really on the radar at that  point.” So this is like a pre-deep-learning   approach. And so the way they did this was they  got a big team of mathematicians and computer   scientists and pathologists and so forth to get  together and build these ideas for features,   like relationships between epithelial nuclear  neighbors.
Thousands and thousands actually they   created of features, and each one required a lot  of expertise from a cross-disciplinary group of   experts at Stanford. So this project took years,  and a lot of people, and a lot of code, and a lot   of math. And then once they had all these features  they then fed them into a machine learning   model - in this case, logistic regression, to  predict survival.
As I say it's very successful,   right, but it's not something that I could create  for you in a minute at the start of a course.   The difference with neural networks is neural  networks don't require us to build these features.   They build them for us! And so what actually  happened was, in I think it was 2015,   Matt Zeiler and Rob Fergus took a trained  neural network and they looked inside it   to see what it had learned. So we don't give  it features, we ask it to learn features.  
So when Zeiler and Zeiler looked inside a  neural network, they looked at the actual   weights in the model and they drew a picture of  them. And this was nine of the sets of weights   they found. And this set of weights, for example,  finds diagonal edges. This set of weights finds   yellow to blue gradients. And this set of weights  finds red to green gradients, and so forth, right.  
And then down here are examples of some bits of  photos which closely matched, for example, this   feature detector. And deep learning, I  mean, is deep because we can then take   these features and combine them  to create more advanced features.   So these are some layer two features. So there's  a feature, for example, that finds corners.  
And a feature that finds curves.  And a feature that finds circles.   And here are some examples of bits of pictures  that the circle finder found. And so remember   with a neural net which is the basic function used  in deep learning, we don't have to hand code any   of these or come up with any of these ideas.
You  just start with actually a random neural network   and your feed it examples and you have it  learn to recognize things, and it turns out   that these are the things that it creates for  itself. So you can then combine these features. And when you combine these features it creates a  feature detector, for example, that finds kind of   repeating geometric shapes.
And it creates a  feature detector, for example, that finds kind of   frilly little things, which it looks like is  finding the edges of flowers. And this feature   detector here seems to be finding words. And so  the deeper you get the more sophisticated the   features it can find are. And so you can imagine  that trying to code these things by hand would be,   you know, insanely difficult, and you  wouldn't know even what to encode by hand,   right! So what we're going to learn is how neural  networks do this automatically, right, but this   is the key difference of why we can now do things  that previously we just didn't even conceive of as  
possible, because now we don't have to hand code  the features we look for. They can all be learned. it's important to recognize we're going to  be spending some time learning about building   image based algorithms and image-based algorithms  are not just for images and in fact this is going   to be a general theme.
We're going to show you  some foundational techniques but with creativity   these foundational techniques can be used very  widely. So for example, an image recognizer   can also be used to classify sounds. So  this was an example from one of our students   who posted on the forum and said for their  project they would try classifying sounds   and so they basically took sounds and  created pictures from their waveforms   and then they used an image recognizer on that  and they got a state-of-the-art result by the way.
Another of our students on the forum  said that they did something very similar   to take time series and turn them into  pictures and then use image classifiers.   Another of our students created pictures  from mouse movements from... from users of   a computer system.
So the clicks became  dots and the movements became lines and   the speed of the movement became colors and  then used that to create an image classifier. So you can see with... with some creativity  there's a lot of things you can do with images. There's something else I wanted to point out which  is that as you saw when we trained a real working   bird recognizer image model we didn't need lots  of math; there wasn't any.
We didn't need lots of   data. We had 200 pictures. We didn't need lots of  expensive computers; we just used my laptop. This   is generally the case for the vast majority of  deep learning that you'll need in... in real life. There will be some math that pops up during this  course but we will teach it to you as needed or   w
e'll refer you to external resources as...  as needed but it'll just be the little bits   that you actually need. You know the myth that  deep learning needs lots of data, I think, is   mainly passed along by big companies that want to  sell you computers to store lots of data and to   process it. We find that most real world projects  don't need extraordinary amounts of data at all   and as you'll see there's  actually a lot of fantastic   places you can do state-of-the-art work for  free nowadays which is... which is great news.
One of the key reasons for this is because  of something called transfer learning   which we'll be learning about a lot during  this course and it's something which   very few people are aware of the pay-off. In this course we'll be using Pytorch. For  those of you who are not particularly close   to the deep learning world, you might have  heard of Tensorflow and not of Pytorch.  
You might be surprised to hear that Tensorflow  has been dying in popularity in recent years   and Pytorch is actually growing rapidly and in…  in research repositories amongst the top papers,   Tensorflow is a tiny minority now  compared to Pytorch. This is also   great research that's come out from Ryan  O'Connor. He also discovered that...
the majority of people that were doing Tensorflow  in 2018 researchers, the majority have now shifted   to Pytorch and I mention this because what people  use in research is a very strong leading indicator   of what's going to happen in industry because  this is where you know all the new algorithms   are going to come out. this is where all  the papers are going to be written about.  
it's going to be increasingly difficult to use  Tensorflow. We've been using Pytorch since before   it came out, before the initial release because  we knew just from technical fundamentals, it was   far better. So this course has  been using Pytorch for a long time.   I will say however that Pytorch requires a lot of  hairy code for relatively simple things.
This is   the code required to implement a particular  optimizer called AdamW in plain Pytorch.   I actually copied this code from the Pytorch  repository so as you can see there's a lot of it. This gray bit here is the code required to do the  same thing with fast.ai. fast.ai is a library we   built on top of Pytorch. This huge difference  is not because Pytorch is bad.
It's because   Pytorch is designed to be a strong foundation  to build things on top of, like fast.ai. So... When you use fast.ai - the library, you get  access to all the power of Pytorch as well but   you shouldn't be writing all this code if you only  need to write this much code, right? The problem   of writing lots of code is that that's lots of  things to make mistakes with, lots of things to,   you know, not have best practices in, lots of  things to maintain.
In general we've found,   particularly with deep learning: less  code is better. Particularly with fastai,   the code you don't write is code that we've  basically found kind of best practices   for you. So when you use the code that we've  provided for you, you know you'll generally   find you get better results. So... so fast.
ai has  been a really popular library and it's very widely   used in industry, in academia, and in teaching  and as we go through this course we'll be seeing   more and more pure Pytorch as we get deeper and  deeper underneath to see exactly how things work. The fast.ai library just won the 2020  best paper award for the paper about it   in Information so again you can see it's  a very well regarded library.
Okay so… Okay we're still green, that's good.   So you may have noticed something interesting,  which is that I'm actually running code   in these slides. That's because these slides  are not in PowerPoint. These slides are in a   Jupyter notebook.
Jupyter notebook is the  environment in which you will be doing   most of your computing.  It's a web-based application   which is extremely popular and widely used  in industry and in academia and in teaching   and it is a very very very powerful way  to to experiment and explore and to build. Nowadays I would say most people at least  most students run jupyter notebooks not on   their own computers particularly for  data science but on a cloud server   of which there's quite a few and as I  mentioned earlier if you go to course.fast.
ai   you can see how to use various  different cloud servers. One I'm going to show an example of is Kaggle.  So Kaggle doesn't just have competitions but it   also has a cloud notebook server and  I've got quite a few examples there. So let me give you a quick example of  how we use Jupyter notebooks. To... to...  
to build stuff, to... to experiment, to explore.  So on kaggle, if you start with somebody else's   notebook... so why don't you start with this one  Jupyter notebook 101. If it's your own notebook   you'll see a button called edit. If it's somebody  else's, that button will say copy and edit.   If you use somebody's notebook that  you like, make sure you click the   upvote button to encourage them  and to help other people find it   before you go ahead and copy and edit. And  once we're in edit mode we can now use this  
notebook and to use it we can type in any  arbitrary expression in python and click run   and the very first time we do that it says  session is starting it's basically launching   a virtual computer for us to  run our code. This is all free. In a sense, it's like the world's most powerful  calculator.
It's a calculator where you have   all of the capabilities of the world's I think  most popular programming language – certainly it   and javascript would be the top two – directly  at your disposal. So, python does know how to   do one plus one, and so you can see here it spits  out the answer. I hate clicking I always use   keyboard shortcuts so instead of clicking this  little arrow, you just press shift enter to do   the same thing but as you can see there's  not just calculations here, there's also   prose, and so jupyter notebooks are great  for explaining to you the version of yourself  
in six months time, what on earth you were  doing, or to your co-workers, or the people   in the open source community, or the people  you're blogging for etc, and so you just type   prose, and as you can see when we create a new  cell, you can create a code cell which is a cell   that lets you type calculations, or a markdown  cell which is a cell that lets you create prose.
And the prose uses formatting in a little mini  language called “markdown”. There's so many   tutorials around I won't explain it to you but  it lets you do things like links and so forth. So I'll let you follow through the tutorial in  your own time because it really explains to you   what to do.
One thing to point out is that  sometimes you'll see me use cells with an   exclamation mark at the start. That's  not Python, that's a bash shell command,   okay? so that's what the exclamation mark means. As you can see you can put images into notebooks,  and so the image I popped in here was the one   showing that Jupyter won the 2017 software  system award which is pretty much the biggest   award there is for this kind of software. Okay,  so that's the basic idea of how we use notebooks.
So let's have a look at how we  do our bird or not bird model. One thing I always like to do when I'm  using something like colab or kaggle cloud,   cloud platforms that I'm not controlling is, make  sure that I'm using the most recent version of any   software.
So my first cell here is exclamation  mark pip install minus u , (that means upgrade)   q (for quiet) fast.ai. So that makes sure  that we have the latest version of fast.ai,   and if you always have that at the start of  your notebooks you're never going to have those   awkward forum threads where you say  “why isn't this working?” and somebody   says to you “oh you're using an  old version of some software!” So, you'll see here this notebook   is the exact thing that I was showing  you at the start of this lesson, So, if you haven't done much python, you might  be surprised about how little code there is here,  
and so python is a concise but not too  concise language. You'll see that there's less   boilerplate than some other languages you might  be familiar with, and I'm also taking advantage   of a lot of libraries so fast.ai provides a lot of  convenient things for you. Oh I forgot to import… So, to use a external library we use  import to “import a symbol from a library”.  
fast.ai has a lot of libraries we  provide. They generally start with   “fast something” so for example to make it  easy to download a url, “fastdownload” has   download_url(). To make it easy to create a  thumbnail, we have Image.to_thumb() and so forth. So, I always like to view – as I'm building a  model – my data at every step.
So that's why   I first of all grab one bird, and then I  grab one forest photo, and I look at them   to make sure they look reasonable. And once I think, “okay they look  okay”, then I go ahead and download. And, so, you can see fast.ai has  a download_images() where you just   provide a list of urls so that's how easy it is,  and it does that in parallel.
So it does that,   you know, surprisingly quickly. One other fast.ai  thing I'm using here is resize_images(). You   generally… you'll find that for computer vision  algorithms you don't need particularly big images,   so I'm resizing these to a maximum size length  of 400 because it just… it's actually much faster   because gpus are so quick for big images, most  of the time can be taken up just opening it.  
The neural net itself often takes less time. So  that's another good reason to make them smaller. Okay, so the main thing I wanted to tell you about  was this data block command. So the data block   is the key thing that you're going to want to get  familiar with as deep learning practitioners at   the start of your journey because the main thing  you're going to be trying to figure out is how do   I get this data into my model? Now that might  surprise you.
You might be thinking we should   be spending all of our time talking about neural  network architectures, and matrix multiplication   and gradients and stuff like that, the truth is  very little of that comes up in practice, and the   reason is, that at this point the deep learning  community has found a reasonably small number of   types of model that work for nearly  all the main applications you'll need;   and fast.ai will create the right type of  model for you the vast majority of the time.  
So all of that stuff about tweaking neural  network architectures and stuff… I mean   we'll get to it eventually in this course  but you might be surprised to discover that   it almost never comes up. Kind of like if you ever  did like a computer science course or something   and they spent all this time on the details of  compilers and operating systems, and then you   get to the real world and you never use it again.
  So this course is called practical deep learning,   and so we're going to focus on the  stuff that is practically important.   Okay, so our images have finished downloading, and  two of them were broken so we just deleted them. Another thing you'll note by the way if you're   a keen software engineer is, I tend  to use a lot of functional style in my   programs I find for kind of the kind of work I do  that a functional style works very well if you're,   you know a lot of people in python are less  familiar with, that it's more, maybe comes,   more from other things. So yeah that's why you'll  see me using stuff like map and stuff quite a lot.
Alright so a data block is the  key thing you need to know about   if you're going to know how to  use different kinds of data sets,   and so these are all of the things,  basically, that you'll be providing.   And so what we did when we designed the data  block was we actually looked and said “okay over   hundreds of projects what are all the things that  change from project to project to get the data   into the right shape”?, and we realized we could  basically split it down into these five things.  
So the first thing that we tell fast.ai is “what  kind of input do we have”?, and so then, so there   are lots of blocks in fast.ai for different kinds  of inputs so we said “ah, the input is an image!;   “what kind of output is there?”, “what  kind of label?.” The output's a category,   so that means it's one of  a number of possibilities.  
So that's enough for fast.ai to know  what kind of model to build for you. So what are the items in this model? what  am I actually going to be looking at to look   to train from? this is a function! In fact  you might have noticed if you were looking   carefully that we use this function here.   It's a function which returns a list of all of  the image files in a path based on extension,   so every time it's going to try and find out  what things to train from, it's going to use   that function. In this case we'll get a list of  image files. Something we'll talk about shortly is  
that it's critical that you put aside some data  for testing the accuracy of your model that's   called a “validation set.” It's so critical that  fast.ai won't let you train a model without one,   so you actually have to tell it  how to create a validation set,   how to set aside some data, and in this case  we say “randomly, set aside 20% of the data.
”   Okay, next question, then you have to tell  fast.ai is “how do we know the correct label   of a photo”, how do we know if it's a bird photo  or a forest photo? and this is another function,   and this function simply returns the parent  folder of of a path and so in this case we   saved our images into either forest or bird.
So  that's where the labels are going to come from And then finally,   most computer vision architectures need all of  your inputs as you train to be the same size,   so “item transforms” are all of the bits  of code that are gonna run on every item,   on every image in this case, and we're saying  okay we want you to resize each of them to being   192 by 192 pixels.
There's two ways  you can resize: you can either crop out   a piece in the middle, or you can squish  it, and so we're saying “squish it”. So that's the data block that's all that you  need, and from there we create an important   class called data loaders. Data loaders are  the things that, actually, pytorch iterates   through to grab a bunch of your data at a time.
  The way it can do it so fast is by using a GPU   which is something that can do thousands of things  at the same time and that means it needs thousands   of things to do at the same time so a data loader  will feed the training algorithm with a “bunch”   of your images at once; in fact we don't call it  a bunch we call it a “batch” or a “mini batch”.
And so, when we say show “batch” that's actually  a very specific word in deep learning. It's   saying show me an example of a batch of data  that you would be passing into the model,   and so you can see showbatch gives you – tells  you – two things: the input which is the picture,   and the label, and remember! the label  came by calling that function.
So,   when you come to building your own models  you'll be wanting to know what kind of splitters   are there? what kinds of labeling functions  are there? and so forth what's wrong button   you'll be wanting to know what kind of labeling  functions are there and what kind of splitters   are there and so forth and so docs.fast.ai is  where you go to get that information.
Often the   best place to go is to choose the tutorials. So  for example here's a whole data block tutorial,   and there's lots and lots of examples, so  hopefully you can start out by finding something   that's similar to what you want to do and see  how we did it, but then of course there's also   the underlying api information  so here's data blocks! Okay. How are we doing? Still doing good! Alright.
So at the end of all this we've got an object  called “dls” that stand for “data loaders”   and that contains iterators that pytorch can  run through to grab batches of randomly split   out training images to train the model with,  and validation images to test the model with. So now we need a model. The critical concept here in fastai is called  a “learner”.
A “learner” is something which   combines a model, which is, that is, the actual  neural network function we’ll be training,   and the data we use to train it with; and  that's why you have to pass in two things:   the data which is the data loaders  object, and a model. And so the model   is going to be the actual neural network function  that you want to pass in, and as I said there's   a relatively small number that basically  work for the vast majority of things you do.
If you pass in just a bare symbol  like this it's going to be one of   fast.ai's built-in models but  what's particularly interesting   is that we integrate a wonderful library by Ross  Wightman called “timm” (the pytorch image models)   which is the largest collection of computer vision  models in the world, and at this point fast.
ai is   the first and only framework to integrate this.  So you can use any one of the pytorch image models   and one of our students Amanamoro was kind  enough to create this fantastic documentation   where you can find out all  about the different models. And if we click on here, you can get lots and lots  of information about all the different models that   Ross has provided.
Having said that,  the model family called “resnet”   are probably going to be fine for nearly all  the things you want to do, but it is fun to   try different models out so you can type in any  string here to use any one of those other models Okay so if we run that, let's see what happens  okay? So this is interesting… So when I ran this…   so, remember on kaggle it's creating a new   virtual computer for us, so it doesn't  really have anything ready to go,   so when I ran this the first thing it did  was it said “downloading resnet18.
pth” What's that? well, the reason we can do this so  fast is, because somebody else has already trained   this model to recognize over 1 million  images of over 1 000 different types;   something called the “ImageNet” dataset, and  they then made those weights available – those   parameters– available on the internet for anybody  to download. By default on fast.
ai when you   ask for a model, we will download those weights  for you, so that you don't start with a random   network that can't do anything. You actually  start with a network that can do an awful lot,   and so, then something that fast.ai has,  that's unique, is this fine_tune method,   which, what it does is: it takes those pre-trained  weights we downloaded for you, and it adjusts   them in a really carefully controlled way  to just teach the model the differences   between your data set, and what it was originally  trained for. That's called “fine tuning”;  
hence the name. So that's why you'll see this  downloading happen first, and so as you can   see at the end of it, this is the error right  here after a few seconds it's a 100% accurate So we now have a learner, and this learner  has been, has started with, a pre-trained   model that's been fine-tuned for the purpose of  recognizing bird pictures from forest pictures.  
So you can now call dot predict on it, and dot  predict you pass in an image, and so this is   how you would then deploy your model. So in  the code you have whatever it needs to do.   So in this particular case this  person had some reason that he needs   the app to check whether they're in the national  park, and whether it's a photo of a bird,   so at the vet where they need to know if  it's a photo of a bird it would just call   this one line of code learn.predict();  and so that's going to return  
whether it's a bird or not as a string,  whether it's a bird or not as an integer, and   the probability that it's a non-bird or a bird;  and so that's why we can print these things out. Okay so that's how we can  create a computer vision model. What about other kinds of models? right? There's  a lot more in the world than just computer vision.  
A lot more than just image recognition.  Well even within computer vision,   there's a lot more than just  image recognition. For example,   for example, there's segmentation! So segmentation, maybe the best way to explain  segmentation is to show you the result of this   model.
Segmentation is where we take photos –  in this case of road scenes – and we color in   every pixel according to what it's a, what is  it. So in this case we've got brown as cars,   blue is fences I guess?  red is buildings? or brown? And so on the left here some photos that  some somebody has already gone through,   and classified every pixel of, every one of these  images according to what that pixel is a pixel of,   and then on the right is what our  model is guessing, and as you can see   it's getting a lot of the pixels correct  and some of them it’s getting wrong.
It's actually amazing how many is getting  correct because this particular model   I trained in about 20 seconds using a tiny, tiny,  tiny, amount of data. So you know, again, like,   you would think this would be a particularly  challenging problem to solve, but it took about   20 seconds of training to solve it, not amazingly  well, but pretty well.
If I trained it for another   two minutes it'd probably be close to perfect. So  this is called “segmentation”. Now you'll see that   there's very, very, little data required, and …  sorry, very little CODE required, and the steps   are actually going to look quite familiar. In fact  in this case we're using an even simpler approach.  
Earlier on, we used “data blocks”. Data blocks  are a kind of intermediate level very … flexible   approach that you can take to handling almost any  kind of data, but for the kinds of data that occur   a lot, you can use these special data loaders  classes which kind of lets you use even less code. So, in this case, to create  data loaders for segmentation,   you can just say: “okay I'm going to pass you  in a function for labeling”; and you can see   here it's got pretty similar things that we pass  in, to what we passed in for data blocks before.
So our file names is get_image_files()  again, and then our label function   is something that grabs this path, and the codes.  So the labels for further segmentation, sorry,   the codes. So, like, “what does each code  mean?” is going to be this text file, but   you can see the basic information we're providing  is very very similar regardless of whether we're   doing segmentation or object recognition; and  then the next steps are pretty much the same.
We create a learner for segmentation. We create,   so, we've got a UNet learner which we'll learn  about later, and then again, we call fine_tune()   so that is it! and that's how  we create a segmentation model! What about stepping away from computer vision?  So perhaps the most widely used kind of   model used in industry is tabular analysis.
So,   taking things like spreadsheets and database  tables, and trying to predict columns of those. So, in tabular analysis, it really looks  very similar to what we've seen already,   We grabbed some data and you'll see when I call  this untar_data(), this is the thing in fast.ai   that downloads some data and decompresses it for  you and there's a whole lot of urls provided by   fast.
ai for all the, kind of common data sets that  you might want to use; you know all the ones that   are in the book? or lots of data sets that are  kind of widely used in learning and research?   So, it makes life nice and easy for you. So  again, we're going to create data loaders,   but this time it's tabular data loaders, but we  provide pretty similar kind of information to what   we have before. Couple of new things we have to  tell it.
“Which of the columns are categorical?”   so they can only take one of a few  values, and “which ones are continuous”   so they can take, basically, any real number,  and then again, we can use the exact same   show_batch() that we've seen before, to  see the data, and so, fast.ai uses a lot of   something called “type dispatch” which is a  system that's particularly popular in in a   language called Julia, to basically,  automatically, do the right thing   for your data, regardless of what kind of data it  is. So, if you call show_batch() on something you  
should go get back something “useful” regardless  of what kind of information you provided.   So for a table it shows you on the information in  that table. This particular data set is a data set   of whether people have less than fifty thousand  dollars or more than fifty thousand dollars in   salary for different districts based on  demographic information in each district.
So, to build a model for that  data loader, we do as always   “something” underscore “learner”.  In this case it's a tabular learner.   Now this time we don't say: “fine_tune()”;  we say “fit”. Specifically, fit_one_cycle().   That's because for tabular models, there's  not generally going to be a pre-trained model   that already does something like what you want,  because every table of data is very different,   whereas pictures often have a similar theme,  you know? They're all pictures. They all have  
the same kind of “general idea” of what  pictures are. So that's why it generally   doesn't make too much sense to fine-tune  a tabular model. So instead, you just fit.   So there's one difference there.  I'll show another example. Okay, so, collaborative filtering. “Collaborative filtering” is the basis  of most recommendation systems today.  
It's a system where we basically take data set  that says: which users liked which products,   or which users used which products, and then we  use that to guess what other products those users   might like based on finding similar users, and  what those similar users like. The interesting   thing about collaborative filtering is that when  we say “similar users,” we're not referring to   similar demographically, but similar in the  sense of people who liked the same kinds of   products. So, for example, if you use any of  the music systems like Spotify, or Apple music,  
or whatever, it'll ask you first, like, “what's  a few pieces of music you like,” and you tell it,   and then it says “okay well maybe let's start  playing this music for you,” and that's how   it works. It uses collaborative filtering. So  we can create a collaborative filtering data   loaders in exactly the same way that we're used  to, by downloading and decompressing some data,   create our collab data loaders.
In  this case we can just save from csv,   and pass in a csv, and this is what collaborative  filtering data looks like. It's going to have,   generally speaking, a user id, some kind of  product id; in this case, a movie, and a rating. So, in this case this user gave  this movie a rating of 3.5 out of 5,   and so, again, you can see show_batch,  right? So use show batch.
You should get back   some useful visualization of your data  regardless of what kind of data it is,   and so again, we create a “learner.”   This time the “collaborative filtering”  learner, and you pass in your data. In this case we give it one extra piece  of information, which is – because this   is not predicting a category, but it's  predicting a real number – we tell it:   “what's the possible range.
”  The actual range is one to five,   but for reasons you'll learn about later, it's  a good idea to actually go from a little bit   lower than the possible minimum, to a little  bit higher. That's why I say 0.5 to 5.5,   and then fine-tune. Now, again, you know we don't  really need to fine-tune here because there's not   really such a thing as a pre-trained collaborative  filtering model.
We could just say “fit”   or “fit_one_cycle”, but actually fine_tune() works  fine as well. So, after we train it for a while,   this here is the “mean squared error”, so it's  basically, that: “on average how far off are we   for the validation set”.  And you can see as we train,   and it's literally so fast (it's  less than a second each epoch)   that error goes down and down, and for any kind of  fast.
ai model you can always call show_results() ,   and get something sensible. So in this case it's  going to show a few examples of users and movies.   Here's the actual rating that user gave that movie  and here's the rating that the model predicted.   Okay, so, apparently a lot of people  on the forum are asking how I'm   turning this notebook into a presentation? So, I'll be delighted to show you, because I'm  very pleased that these people made this thing   for free for us to use. It's called “Rise”,  and all I do is, it's a notebook extension,  
and in your notebook it gives you an extra little  thing on the side where you say which things are   slides, or which things are fragments, and a  fragment just being… so, this is a slide, that's   a fragment. So if I do that you'll see it starts  with a slide and then the fragment gets added in. Yeah, that's about all there is to it!  Actually it's pretty great, and it's very   well documented.
You know I'll just mention,  like what do I make with Jupyter notebooks:   this entire book was written  entirely in Jupyter notebooks. Here are the notebooks. So if you go to the  fastai fastbook repo… if you go to the fastai   fastbook repo, you can read the whole book, and  because it's all in notebooks, every time we say:   “here's how you create this plot,” or “here's  how you train this model,” you can actually   create the plot, or you can actually train  the model, because it's all notebooks.
The entire fast.ai library is  actually written in notebooks.   So you might be surprised to discover  that if you go to fast.ai fast.ai   that the source code for the  entire library is notebooks,   and so the nice thing about this  is that, you know, the source code   for the fast.ai library has actual pictures of  the actual things that we're building for example.
What else have we done with notebooks? oh! blogging! I love blogging with notebooks, because  when I want to explain something   I just write the code, and you can just  see the outputs, and it all just works.   Another thing you might be surprised by, is  all of our tests and continuous integration   are also all in notebooks.
So every  time we change one of our notebooks…   every time we change one of our notebooks,  hundreds of tests get run automatically,   in parallel, and if there's any issues  we will find out about it. So, yeah,   notebooks are great! and rise is… is a  really nice way to do slides in notebooks. Alright. So, what can deep learning do at present?   We're still scratching this the tip of the iceberg  even though it's a pretty well hyped you know   heavily marketed technology this pro… you  know when we started in 2014 or so, you know,   not many people were talking about deep learning,  and really there was no accessible way to get  
started with it. There were no pre-trained  models you could download, you know. There   was just starting to appear some of the first  open source software that would run on gpus,   but yeah, I mean, but despite the fact that  today there's a lot of people talking about deep   learning, we're just scratching the surface.
  Every time pretty much somebody says to me   I work in domain x, and I thought I might  try deep learning out to see if it can help,   and I see them a few months later,  and I say “how did it go?” they   nearly always say: “well we just broke the  state-of-the-art results in our field.”   So, you know, when I say these are things that  it's currently state of the art for – these are   kind of the ones that people have tried so far  – but still, most things haven't been tried.
So, in NLP, deep learning is the state of the art   method in all these kinds of things and a  lot more; computer vision, medicine, biology   recommendation systems, playing games, robotics…  I mean it's just… I've tried… I've tried elsewhere   to make bigger lists, and I just end up with  pages and pages and pages! so yeah it's… it's   generally speaking you know, if it's something  that a human can do reasonably quickly,   like look at a go board and  decide if it looks like a good   go board or not, even if it needs to be an  ex even… if it needs to be an expert human,  
then that's probably something that  deep learning will be pretty good at.   If it's something that takes a lot of logical  thought processes over an extended period of time,   you know, particularly if it's not based  on much data, maybe not. You know, like,   “who's going to win the next election”, or  something like that.
That'd be kind of broadly   how I try to decide: is your thing useful for  deep… you know, good for deep learning or not.   You know it's been a long  time to get to this point. Yes, deep learning is incredibly powerful now,  but it's taken decades of work. This was the first   neural network. Remember, neural networks are the  basis of deep learning. So this was back in 1957.  
The basic ideas have not changed much at all,  but you know we do have things like gpus now,   and solid-state drives, and stuff like that,  and of course much more data, just is available,   now, but this has been decades of really hard  work by a lot of people to get to this point. So let's kind of take a step back, and  talk about, like, what's what's going on   in these models, and I'm going to describe  the basic idea of machine learning   largely as it was described by Arthur  Samuel in the late 50s when it was invented,  
and I'm going to kind of do it with  these graphs; oh which, by the way,   you might find fun… these graphs are themselves…   created with Jupyter notebooks. So these are  graphviz descriptions that are going to get   turned into these, so there's a little  sneak peek behind the scenes for you? So, let's start with kind of a graph of, like,  well, what does a normal program look like? right?   so, in the pre deep learning, and machine learning  days, well, you know, you'd have… you still have   inputs, and you still have results. Right? and  then you code a program in the middle, which is,  
you know, a bunch of conditionals, and loops,  and setting variables, and blah blah blah. Okay.   A “machine learning model”  doesn't look that different… but… The program has been replaced with something  called “a model,” and we don't just have inputs   now, we now also have weights, which are also  called “parameters,” and the key thing is this:   the model is not any more a bunch of  conditionals, and loops, and things.
It's a mathematical function. In the case of a neural network, it's a  mathematical function that takes the inputs,   multiplies them together by the weights –  by one set of weights – and adds them up;   and then it does that again for a  second set of weights and adds them up,   does it again for a third set of  weights, and adds them up and so forth.
It then takes all the negative  numbers, and replaces them with zeros,   and then it takes those as inputs to  the next layer. It does the same thing;   multiplies them a bunch of times, and adds  them up, and it does that a few times,   and that's called “a neural  network.” Now, the model   therefore, is not going to do anything useful,  unless these weights are very carefully chosen.
And, so the way it works is, that we actually  start out with these weights as being   random. So initially, this thing  doesn't do anything useful at all! So, what we do –- the way arthur samuel  described it back in the late 50s   (the inventor of machine learning) – is,  he said: “okay, let's take the inputs,   and the weights, put them through our  model.
” He wasn't talking particularly   about neural networks. He's just  like… whatever model you like… get the results, and then let's decide how  “good” they are, right? So, if for example,   we're trying to decide: “is this a picture of  a bird?” and the model said – which initially   is random – says “this isn't a bird,” and actually  it IS a bird, we would say: “oh you're wrong!” So   we then calculate “the loss.” So, the loss is  a number that says how good were the results.
So that's all pretty straightforward, you  know. We could, for example, say oh what's   the accuracy? We could look at 100 photos and  say which percentage of them did it get right.   No worries. Now the critical step is this  arrow. We need a way of updating the weights   that is coming up with a new set of weights  that are a bit better than the previous set.  
Okay. And by a bit better we  mean it should make the loss   get a little bit better. So we've got this  number that says how good is our model and   initially it's terrible, right? It's random. We  need some mechanism of making a little bit better. If we can just do that one thing   then we just need to iterate this a few times.
  Because each time we put in some more inputs   and... and put in our weights and get our loss  and use it to make it a little bit better,   then if we make it a little bit better enough  times, eventually it's going to get good.   Assuming that our model is flexible enough to  represent the thing we want to do. Now remember   what I told you earlier about what a neural  network is: which is basically multiplying things   together and adding them up and replacing the  negatives with zeros and you do that a few times,   that is provably an infinitely flexible function.
So it actually turns out that that incredibly  simple sequence of steps, if you repeat it a   few times, you do enough of them, can solve any  computable function. And something like generate   an artwork based off somebody's twitter bio  is an example of a computable function, right?   Or translate English to Chinese is an example  of a computable function.
So they're not the   kinds of normal functions you do in year eight  math right but they are computable functions.   and so therefore if we can just create this step  then and use the neural network as our model   then we're good to go. In theory we can solve  anything given enough time and enough data   and so that's exactly what we do.
And so once we've finished that training procedure   we don't need the loss anymore and even the  weights themselves, we can integrate them,   kind of into the model, right? We finish  changing them so we can just say that's now fixed   and so once we've done that we now  have something which takes inputs,   puts them through a model and gives us results.
It  looks exactly like our original idea of a program   and that's why we can do what I described earlier,  that is once we've got that learn.predict for our   bird recognizer we can insert it into any piece  of computer code, right? Once we've got a trained   model it's just another piece of code we can  call with some inputs and get some outputs. Deploying machine learning models in practice  can come with a lot of, you know, little tricky   details but the basic idea in your code is that  you're just going to have a line of code that says   learn.predict and then you just fit it in with  all the rest of your code in the usual way.  
And this is why: because a trained model is  just another thing that maps inputs to results. Okay. All right. so as we come to  wrap up this first lesson,   for those of you that are already familiar with  notebooks and Python, there's... you know... this   has got to be pretty easy for you.
You're just  going to be using some stuff that you're already   familiar with and some slightly new libraries.  For those of you who are not familiar with Python,   you know, you... you're biting into a big thing  here. There's obviously a lot you're going to   have to learn and to be clear I'm... I'm not  going to be teaching Python in this course   but we do have links to great python resources in  the forum. So check out... check out that thread.  
Regardless of where you're at, the most important  thing is to experiment and so experimenting   could be as simple as just running those Kaggle  notebooks that I've shown you just to see them   run. You could try changing things a little bit.  I'd really love you to try doing the... the bird   or forest exercise but come up with something  else.
Maybe try to use three or four categories   rather than two, you know. Have a think about  something that you think would be fun to try. Depending on where you're at, you know, push  yourself a little bit but not too much. So make   sure you get something finished before the next  lesson. Most importantly, read chapter one of the   book. It's got much the same stuff that we've seen  today but presented in a slightly different way.  
And then come back to the forums and present  what you've done in the share your work here   thread. After the first time we  did this in year one of the course   we got over a thousand replies and of  those replies it's amazing how many   of them have ended up turning into new  startups, scientific papers, job offers.
It's been really cool to watch people's journeys  and some of them are just plain fun, you know. So   this person classified different types of  Trinidad and Tobago people. So you know,   people do stuff based on where they  live and what their interests are.   I don't know if this person is particularly  interested in zucchini and cucumber but they   made a zucchini and cucumber classifier.
I  thought this was a really interesting one:   classifying satellite imagery into what city  it's probably a picture of. Amazingly accurate   actually. 85 percent with 110 classes. Panama  city bus classifier. Batik cloth classifier.   This one, you know, very practically important,  you know, recognizing the state of buildings.   We've had quite a few students actually move into  disaster resilience based on satellite imagery   u
sing exactly this kind of work. We've already  actually seen this example. Ethan Sutin the... the   sound classifier and I mentioned it was  state of the art. He actually checked up the   dataset's website and found that he beat the state  of the art for that. Alena Harley did tumor-normal   sequencing. So she was at Human Longevity  International. So she actually did three   different really interesting pieces of cancer work  during that first course if I remember correctly.  
And I showed you this picture before. What I  didn't mention is actually this... this student   Gleb was a software developer at Splunk, a big  NASDAQ listed company. And this student project   he did turned into a new patented product at  Splunk and a big blog post and the whole thing   turned out to be really cool.
Basically something  to identify fraudsters using image recognition   with these pictures we discussed. One of our  students built this startup called Envision. Anyway there's been lots and lots of examples. So  all of this is to say, you know, have... have a go   at you know, starting something. Create something  you think would be fun or or interesting and share   i
t in the forum. If you're a total beginner  with Python, you know, then start... start   with something simple. But I think you'll find  people very encouraging and if you've done this   a few times before then try to push yourself  a little bit further. And don't forget to look   at the quiz questions at the end of the book  and see if you can answer them all correctly. All right, thanks everybody, so much for coming.  Okay. Thanks so much for coming everybody. Bye.

---

# Lesson 2: Practical Deep Learning for Coders 2022 - YouTube
https://www.youtube.com/watch?v=F4tvM4Vb3A0

Transcript:
Hi everybody. Welcome to lesson two. Thanks for  coming back… slight change of environment here, we had a bit of an “administrative issue” at our  university — somebody booked our room — so I'm doing this from the study at home. so sorry  about the lack of decorations behind me. I'm actually really really pumped about this  lesson.
It feels like going back to what things were like in the very early days, because we're  doing some really new, really cool stuff, which… you know… stuff that hasn't really been in courses  like this before. So, I'm super super excited. So thanks a lot for coming back after lesson  one, and I hope it's worth you coming back. I think… I think you're gonna love it! I  am, yeah, I'm really excited about this.
Now remember that, you know, the course  goes with the book, so be sure that you're following. I mean, not following along  in the book, because we're covering similar things in different directions… but, like,  read the book as well, and remember the book is entirely available for free as well. You can go  to the fast.
ai fastbook repo to see the notebooks, or through course.fast.ai you can read  it there — for example, through Colab. And also remember that the book… I mean… the  book has a lot of stuff that we didn't cover in the course like, you know, stuff I find pretty  interesting about the history of neural networks, some of which has some really interesting  personal stories actually, as you'll read here.
And at the end of each chapter there is a quiz.  Remember it's not a bad idea before you watch the video to read the quiz, so if you  want to read the Chapter Two quiz, you know, and then come back, that's not  a bad idea. Then make sure that you can do the quiz after you've watched the video,  and you've read chapter two of the book.
Something I didn't mention last week  is there's also a very cool thing that Radek — who I mentioned last week  — has written, called aiquizzes.com which is a site specifically  for quizzes about the book. And it actually uses repetitive based learning  techniques to make sure that you never forget. So, do check out aiquizzes.com.
It's all brand  new questions, they're different to the ones in the book, and they're really nicely curated, and  put together. So check out aiqquizzes.com as well. Remember, as well as course.fast.ai, there's also forums.fast.ai. So course.fast.ai  is where you want to go to get, you know, links to all the notebooks, and kaggle stuff, and  all that stuff. You'll also find on forums.fast.
ai every lesson has an official topic,  you know, with all the information you'll need. Generally there'll be a bit more  info on the forums, we try to keep the course lean and mean, and the forums are a bit more  detailed. So if you find — in this case you are gonna look for the Lesson Two official topic  but here's the Lesson One official topic so far.
So from the Lesson One official topic, already, after just a few days since I recorded it  — we haven't even launched the course so it's just the people doing it live — there's already a lot  of replies, and that can get pretty overwhelming. So be aware that there's a button at the bottom  of my post that says summarize this topic.
And if you hit that, then you'll just see the  most upvoted replies, and that's a really good way to just make sure that you hit on the main  stuff. So there's the button, and here's what it looks like after you hit it. You'll just get…  you'll just get the upvoted stuff from fast.ai legends like Sanyam and Tanishq. So hopefully  you'll find that a useful way to use the forum.
So one of the cool things about this week is I…  as promised, put up the “show us what you've made” post, and already a lot of people have posted.  I took this screenshot a few days ago, it's way above 39 replies already, if I'm… if I remember  correctly. I had a lot of trouble deciding which ones to share because they're all so good, so I've  actually decided to kind of, you know, go the easy route, and I just picked the firsts.
So I'm just  going to show you the first ones that were posted because they're all so good. So the first, the  very very first one to be posted is a damaged car classifier… so that worked out pretty well  — it looks like. And I really liked what Matt, the creator, said about this, is that… you know,  wow, it's a bit uncomfortable to run this code, I don't really understand yet but I'm just doing it.
  And so I'm like, yeah, good on you Matt for just… for just doing it. That's the way to get started,  it's all going to make sense, don't worry. Very nice to see that the next one posted was  actually a blog post in fast pages — very nice to see — so just describing some stuff, some  experiments that they ran over the week and what did they find.
Next one was the amazing beard  detector which if I understand correctly was mainly because it's very easy to get from bird to  beard by just changing one letter to two, and this is doing a very good job of finding gentlemen  with beards. Very nice. And then this one is… another level again, it's a whole in-production  web app to classify food, which is kind of like extra credit . Apparently we're up to 80 replies  now in that thread, thank you Sanyam, very cool.
So, you know, obviously… so this was actually  created by Suvash who's been doing the courses for a few years now, I believe. And so you know, one  day you too might be able to create your very own web app and put it in production. And when I say  one day, more specifically… today! I'm actually going to show you how to do this right now! So  it's actually quite a lucky coincidence that Suvash put this up there because it's exactly  the topic that we're going to pick today.
So, how do we go about putting a  model in production? Step one, is… well… you've kinda done step one, right? Step one  is… step one, two, three, four… is to figure out what problem you want to solve, figure out how  to find the data for it, go gather some data, and so forth.
So what's the kind of first step  after you've got your data? The next step is data cleaning, and if you go to Chapter 2 of the  book, which I'm going to go ahead and open up now… So here is the book… so you can open it in  Colab directly from the course, or if you've cloned it to your computer, or whatever,  you can… you can do it there. So remember, course.fast.
ai will run you through exactly how to  run these notebooks, and so you can see Chapter 2 is all about putting stuff in  production, and so here is Chapter 2. All right, and so remember we hit Shift +  Enter to run cells, okay? …to execute them. And so we're going to go to the part of  the book where we start cleaning the data. So I'll click on Navigate, and we'll go  down here… gathering data… there we are! So we could do a quick bit of revision first.
By  the way, I will mention, a lot of people ask me what are the little tricks I use for getting  around a Jupyter notebook so quickly and easily. One of the really nice ones as you'll see is  this navigate menu, which actually doesn't appear by default. So if you install something  called Jupyter Notebook Extensions - Jupyter… Notebook… Extensions… and so you just… you just  “pip install” them, follow the instructions and then restart Jupyter.
Obviously this… Colab  already has a table of contents by the way so this is just if you're using something  local, for example, then you'll see here that this “Nbextensions” thing will appear,  and if you click on “Table of Contents (2)” that gives you this handy navigation  bar. The other thing I really like is this one here called collapsible headings, and  that's the one which gives me these nice little things here, to close, and open up.
  And actually that's not even the best part. The best part for me is if I hit the  right arrow it goes to the end of a section, and if I hit the left arrow it goes to  the start of a section. So it's like, if I want to move around sections I just press  up left, down right, down right, very handy, and if you hit left again when you're here  it 'll close it up.
Hit right again here, open it up. So that's collapsible headings.  Anyway, a couple of really handy things, and we'll be talking a lot more about getting  your notebook set up today in a moment. Okay, so one thing you'll notice is in the book,  we use the Bing API for searching images. I've just gone ahead, and replaced “bing” with “ddg”  because the Bing API requires getting an SDK key which honestly, it’s like the hardest thing in  deep learning is figuring out the Bing Azure website, and getting that sorted out. ddg doesn't!  So it's basically exactly the same… (and you can…)
I'll share this notebook as well, on the course  website in the forum but all I've basically done is I've replaced bing with ddg, and got rid of  the key. So then, just like we did last week, we can search for things, and so in the book we  did a bear detector because at the time I wrote it my then toddler was very interested  in me helping identify teddy bears, and I certainly didn't want her  accidentally cuddling a grizzly bear, so we show how here how we can search  for grizzly bears, just like last week.
Something that loops through grizzly  bears, black bears, and teddy bears, just like last week. Get rid of the  ones that failed, just like last week. And one thing a few people have asked on the  forum is how do i find out more information about basically any python or fastai or pytorch  thing. There's a few tips here in the book.
One is that if you put a double question mark next to  any function name you'll actually get the whole source code for it, and by the same token if you  put a single question mark you'll get a brief… you know… a little bit of  information. If you've got nbdev installed (I think it's nbdev  you need) then you can type doc, and that will give you perhaps most importantly  a link straight to the documentation where you can find out more information and  generally there will be examples as well.
And also a link here to the  source code, if you want to… (let's do that with a control okay) …a link  to the source code and that way you can jump around. And notice that in github in the source  code you can click on things and jump to their definition. So it's kind of a nice way of skipping  around to understand exactly what's going on.
Okay, so lots of great ways of getting help. But what I promised you is that we're going to  now clean the data. So I'm going to tell you something that you might find really surprising.  Before you clean the data, you train a model. Now, I know that's going to sound really backwards  to what you've probably heard a thousand times, which is that first you… ~(build/you/train)  …you clean your data and then you train your model. But I'm going to show you something really  amazing. First we're going to train a model and
you'll see why in a moment. So to train a model,  just like before, we use a DataBlock to grab our data loaders — there's lots of information  here in the book about what's going on here. There we go… and so then we can call  show_batch to see them as per usual. There's a little side bar here in the book, I'll  quickly mention, which is about the different ways we can resize.
I think we briefly  mentioned it last week… we can squish… last week I used a string, you can use a string  or this kind of enum like thing that we have. You can see with a squish you can  end up with some very thin bears, right? So this is the real size, the shape, of  the bear. Here its become thin but you can see now we've got all of its cubs… (are they called  clubs? yeah bear cubs) …so it squished it to make sure we can see the whole picture. Same one  here.
This one was out of the picture, we squished it and this guy now looks weirdly thin, but we  can see the whole thing. So that's squishing! Or else this one here is cropping. It's cropped  out just the center of the image so we get a better aspect ratio but we lose some stuff. This  is so we can get square images. And the other approach is we can use “pad.” And so you can pad  with various different things.
If you pad with zeros, which is black, you can see here now we've  got the whole image and the correct aspect ratio. So that's another way we can do it and you  know, different situations, you know, result in different quality models. You can try them, all  it doesn't normally make too big a difference. So I wouldn't worry about it too much.
I'll  tell you one though that is very interesting is RandomResizedCrop. So instead of saying resize,  we can say RandomResizedCrop. And if we do that you'll see we get a different bit of an image  every time. So, during the week… this week somebody asked on the forum… I'm trying  to… (this is a really interesting idea, which it turned out worked slightly) they  wanted to recognize pictures of French and German texts.
So obviously this is not the  normal way you would do that, but just for a bit of experiment (and I love experiments,  so) they had very big scans of documents and they wanted to figure out whether it was  french or german just by looking at images and they said the pictures were too big what should  I do? I said, use RandomResizedCrop and that way you'd grab different bits of the image, and this  is very nice because you could run lots and lots of epochs and get slightly different pictures  each time. So this is a very good technique.
And this idea of getting different pictures each time  from the same image is called “data augmentation.” And again, I'm not going to go into too  much detail about data augmentation, because it's in the book, but  i'll just quickly, you know, point out here that if you use this thing called  “aug_transforms” so augmentation transforms, and here I have multiplied them by two  – so I've made them super big so you can see them more easily. You can see that these  teddies are getting turned, and squished, and
warped, and recolored, and saturated, all this  stuff to make every picture different. And generally speaking if you're training  for more than about five or ten epochs, which you'll probably want to do most of the time  unless you've got a super easy problem to solve, you'll probably want to use RandomResizedCrop  and these “aug_transforms”.
Don't put the mult=2, just leave that empty. I'm just putting  it there so you can see them more clearly. So I've got an interesting question here from  Alex in our audience, which is: “Is this… is this copying the image multiple times? …doing  something like this? …or something like this?” and the answer is “no we're not copying the image.
  What happens is that image… so each epoch, every image gets read. And what happens here is though…  is… is kind of… in memory, in RAM, this… the image is being “warped”, right? it's being… we're  cropping it and recoloring it, and so forth. So it's a real time process that's happening during  model training. So, there's no copies being stored on your computer, but effectively it's almost  like there's infinitely slightly different copies because that's what the model ends up seeing.
So,  I hope that makes sense Alex, and everybody else, so that's a great question! Okay, so we've got… we're going to use  RandomResizeCrop, we're going to use augmentation transforms, so that we can get at DataLoaders from  that and then we can go ahead and train our model. It takes about a minute.
In this case we only did  four epochs of fine tuning, and we'll talk about why there's five here later in the course,  but four main epochs of fine tuning. So we probably didn't really need RandomResizedCrop and  aug_transforms because there's so few epochs, but you know, if you want to run more epochs  this is a good approach. Under three percent error! that's good! Okay, so remember I said  we're going to train a model before we clean? Okay so let's go ahead and train it.
So while that's training… so it's running on my  laptop, which only has a 4GB GPU, it's pretty basic, but it's enough to get started. While  that's training we'll take a look at the next one. So the first thing we're going to look at is  the confusion matrix and the confusion matrix is something that… it only is meaningful  for when your labels are categories, right! And what it says is how/what category errors are  you making.
And so this is showing that the model that we've got at this point… there were two  times when there was actually a grizzly bear and it thought it was a black bear, and there was  two times when there was actually a black bear, and it thought it was a grizzly bear, and  there was no times that it got teddies wrong, which makes sense right because teddies  do look quite different to both.
In a lot of situations when you look at this,  it'll kind of give you a real sense of, like okay, well what are the hard ones? right? So for example  if you use the pets data set that we quite often play with in the book and the course, this  classification [**] matrix for different breeds of pets, you know, really shows you which ones are  difficult to identify, and I've actually gone in, and like, read wikipedia pages, and pet  breeding reports about how to identify these particular types, because they're so  difficult, and even experts find it difficult.
And one of the things I've learned from doing  the course actually is… black bears and grizzly bears are much harder to pick apart than I  had realized... so I'm not even going to try! But I'll show you, the really interesting  thing we can do with this model, is that now we've created this  classification interpretation object, which we use for confusion  matrix, we can say plot_top_losses.
We can say plot top losses, and this  is very interesting. What it does is, it tells us the places where the “loss” is the  highest. Now if you remember from the last lesson, the loss is that measurement – of how good our  model is – that we take after each time we run through an item of data.
A loss will be “bad”  if we predict wrongly, and we're very confident about that prediction. So here's an example  where we predicted… (here's the order here: prediction, actual loss, probability) …where we  predicted grizzly and it was actually a black, and we were 96% sure (our model  was) that it's a grizzly. Now, I don't know enough about bears to know whether  the model made a mistake or whether this actually is a picture of a grizzly bear… but so, an expert  would obviously go back and check those out… right? Now, you'll notice a couple here... it's  got grizzly/grizzly, teddy/teddy… they're actually
correct, right? So why is this loss “bad”? When  it's correct? And the reason is… because it wasn't very confident. It was only 66% confident,  right? So here's a teddy… it's only 72% confident right? So you can have a bad loss, either by  being wrong and confident, or being right and unconfident.
Now the reason that's really helpful,  is that now we can use something called the fast.ai ImageClassifierCleaner to clean up the  ones that are wrongly labeled in our data set. So when we use the ImageClassifierCleaner it  actually runs our models. That's why we pass it “learn”, right? And I mentioned that I don't know  much about black bears and grizzly bears but I do know a lot about teddy bears, so I'll pick teddy  bears.
And if I click teddy bears it's now showing me all the things in the training set (you can  pick training or valid) that were marked as teddy bears and here's what's really important: they're  ordered by loss, so they're ordered by confidence, right? So I can scroll through just the first  few and check they're correct, right? And oh, here's a mistake, right? So when I find  one that was wrongly gathered I can either, put it…if it's in the wrong category,  I can choose the correct category… or, if it shouldn't be there at all, I click delete.  So here I'll go ahead and click delete, right?
So you can see some reasons that some of these  are hard, like for example here's two teddies, which is just, I guess, confusing since  it doesn't see that often. This one here is a bit weird looking, it looks almost like  a wombat. This is an awful lot of teddies. This one maybe is just a bit hard to see from the  background, but these, otherwise, they look fine… Fine, so we just look through the first  few and if you don't see any problem or problems in the first year, you're probably  fine. So that's cleaned up our training set,
let's clean up our validation set as well.  So here's that one it had trouble with, I don't know why it had trouble with that one but  so be it and we'll have a quick scroll through… Okay I'm not really sure that's a bear so  I'm just going to go ahead and delete it. It's a teddy something… but, you know, it's  a problem.
Okay, that's not a teddy either… So you see the idea, right? So after we've  done that, what that does is… the cleaner has now stored a list of the ones that we  changed and the list of the ones we deleted, so we can now go ahead and run this cell. And so  that's going to go through a list of all of the indexes that we said to delete, and it will  delete those files.
And we'll go through all the ones we set to change, and it will move  them to the new folder, there we go, done! So, this is like, not just something for  image models. It's just… it's actually… a really powerful technique that  almost nobody knows about and uses, which is, before you start data cleaning,  always build a model to find out what things are difficult to recognize in your  data and to find the things that the model can help you find data problems.
And then as  you see them, you'll kind of say “okay, I see the kinds of problems we're having” and  you might find better ways to gather the next data set, or you might find ways to kind of  automate some of the cleaning and so forth. Okay, so, that is data cleaning and since I  only have a 4GB GPU it's very important for me to close and halt, because that will free  up the memory.
So it's important to know… on your computer your normal RAM doesn't really  get filled up, because if you use up too much RAM, what will happen is that instead your  computer will start (it's called) swapping, which is basically to save that RAM onto the  hard disk to use it later. GPUs can't swap. GPUs, when they run out of RAM, that's it,  you're done.
So you need to make sure that you close any notebooks that are using the  GPU that you're not using and really only use one thing at a time on the GPU, otherwise  you're almost certainly run out of memory. So we've got the first few reds starting  to appear. So remember to ask and, in terms of the yellows, it's important to  know, as you watch the video, I'm not asking you to run all this code, okay? The idea is to  kind of watch it, and then go back and pause, you know, as you go along. Or you can just stop,  try, stop, try. The approach I really like — and a
lot of students really like — for watching these  videos is to actually watch the entire thing without touching the keyboard, to get a sense of  what the video is about, and then go back to the start and watch it again and follow along. That  way, at every point you know what it is you're doing, you know what's going to happen next.  That can actually save you some time.
It's a bit of an unusual way because, obviously like,  real life lectures you can't do that, you can't rewind the professor and get them to say it  again, but it's a good way to do it here. So now that we've cleaned our data, how are  we going to put it into production? Well in the book we use something called Voilà.
  And it's… it's pretty good! But there's actually something that I think most of you are  probably going to find a lot more useful nowadays, which is something called HuggingFace  Spaces, and there's a couple of things you can use with that. We're going to  look at something called Gradio today. And there isn't a chapter about this in the  book but that doesn't matter because Tanishq Abraham — who's actually one of the TAs in the  course — has written a fantastic blog post about really everything we're going to cover today. So  there's a link to that from the forum and from the
course page, so this is like the equivalent  of the chapter of the book, if you like. And I would be remiss if I didn't stop for  a moment and call out Tanishq in a big way for two reasons: the first is he is one of the  most helpful people in the fastai community, he's been around quite a few years,  incredibly tenacious, thoughtful and patient; and also because I have this fantastic picture  of him a few years ago with Conan when he was a famous child prodigy, so now you know what  happens to famous child prodigies when they
grow up – they became even more famous fastai  community members and deep learning experts. So you should definitely check out this  video of him telling jokes to Conan. I think he's still only 18 actually,  this is probably not that many years ago. So thank you very much Tanishq  for all your help in the community and sorry for embarrassing you with that picture  of you as a nine-year-old — I'm not really, haha.
Okay, now, the thing is for doing  Gradio and HuggingFace Spaces, well, it's easy enough to start. Okay, we start over  here on the HuggingFace Spaces page — which we've linked to from the forum in the course.  And we're going to put a model in production, where we're going to take the model we  trained, and we are going to basically copy it to this HuggingFace Spaces server  and write a user interface for it.
So… let's go! Create new space. Okay, so, you can  just go ahead and say, all right… so obviously you sign up… the whole thing's free. Basically  everything I'm showing you in this entire course, you can do for free, that's the good news. Okay,  so, give it a name, just create something minimal. I always use the Apache license because  it means other people can use your work really easily but you don't have  to worry too much about patents.
As I say there's a few different products you can  use with it, we're going to use Gradio, also free. If you make it public then you can share it,  which is always a good idea when you're a student, particularly to… to really be building up that  portfolio. Okay, so we're done. We've created a space.
Now, what do we do next? Well, Spaces  works through Git, now, most software developers will be very familiar with Git. Some data  scientists might not be. And so Git is a very, very useful tool. I'm not going to talk about it  in detail, but let's kind of quickly learn about how to use it, right? Now, Git, you can use  it through something called Github Desktop which is actually pretty great and even people  who use Git through the console should probably be considering using Github Desktop as well, because  some things are just much faster and easier in it, in fact I was talking to my friend Hamil today and  I was like, oh, help, I've accidentally committed
this two things by mistake, what's the easiest  way to revert it, and he used to work at Github and I thought he was going to have some fancy  console command and he's like “oh, you should use Github Desktop and you can just click on it.”  Oh! That's a great idea. So that's… that's useful. But most of the time we do use Git from  the console, from the terminal.
If you're a Linux user or a Mac user you've already got  a terminal, very straightforward, no worries. If you're a Windows user I've got good news,  nowadays Windows has a terrific terminal. It's called Windows Terminal, you  get it from the Microsoft Store, so in fact every time you see me using a terminal,  I'm actually using that Windows Terminal, works very well. God knows why you'd want it to, with  all these ridiculous colors but, there you go.
Now, what do you want to be running inside your  terminal? Obviously, if you're in Linux or Mac you've already got a shell set up. In Windows, you  almost certainly want to use Ubuntu. So Windows, believe it or not, can actually run a full Linux  environment and to do it, is right typing a single line, which is this… So, if you go to (just google  for WSL install) run PowerShell as administrator, paste that command, wait about  five minutes reboot, you're done.
You now have a complete Linux environment.  Now, one of the reasons I'm mentioning this is, I'm going to show you how to do  stuff on your own machine now, and, so, this is like going to a bit of an  extra level of geekery, which some data scientists may be less familiar with.
So you  know, don't be worried about the terminal, you're gonna.. I think you're gonna… find it really  helpful and much less scary than you expect. And, I particularly say, like for me, I choose to  use Windows and that's because I get, you know, all the nice Windows GUI apps and I can draw on  my screen, and do presentations, and I have a complete Linux environment as well.
And that Linux  environment uses my GPU and everything, so for me, you know, my first choice is to  use Windows, my second choice, by not very much, really like it, would be  to use Linux. Mac is a little bit harder, but it's still usable. Some things are a little  bit trickier on Mac but, you should be fine. Okay, so, whatever you've got, at this  point you've now got a terminal available.
And so, in your terminal… (one of  the really nice things about using a terminal is you don't have to follow lots  of instructions about click here, click here, click here. You just copy and paste things.)  So I'm just going to… you just copy this, and you go over to your terminal, and you paste  it in, and you run it, and after you do that, you'll find that you've now got a directory.
  And so that new directory initially is empty and they tell you, okay, go ahead and create  a file with this in it. Okay, so, how do you create a file with that in it, when we're in  here, in our Linux environment on Windows, or in the terminal on Mac, or whatever? Well, all  you do in Windows, if you just type “explorer.exe .
”, it'll open up explorer here, or better  still, on either Mac, or Linux, or Windows… So, yeah, regardless of what computer (type  of computer on), you can just type “code .” and it will pop up Visual Studio Code and open  up your folder. And so then you can just go ahead and… (if you haven't used VS Code before it's  really well worth taking a few minutes to read some tutorials, it's a really great IDE) …and  so you can go ahead and create an app.
py file, like they tell you to, app.py file, containing  what they told you to put in it. Here it is, here. All right, we're nearly there. So  you can now go ahead and save that, and then you need to commit it, to Gradio…  (not Gradio…) to HuggingFace Spaces. So one really easy way is just, in Visual Studio  itself, you can just click here, and that'll give you a place where you type a message,  and you hit tick, and it'll send it off to HuggingFace Spaces for you.
  So once you've done that, you can then go to… back to… the  exact same website you're in before: “huggingface.co/spaces/jhp00/minimal”, and what you'll find now is that — it'll take  about a minute, to build your website. And it's… the website it's building is going to have a  Gradio interface with a text input, a text output, and it's going to run a function  called “greet()” on the input, and my function called “greet()” will return  “Hello {name}”. So that's what it's going to do.
Now there it goes, let's try  it! We'll say hello to Tanishq. I'm not always very good at remembering how  to spell his name, I think it's like that. And there you go, so you can see it's  put the output, for our input. So not a very exciting app but we now have, to be  fair, an app running in production, right? Now, I told you we'd have a deep  learning model running in production so now we have to take the next step which  is to turn this into a deep learning model.
All right, so, first we're going  to need a deep learning model. And there's a few different ways we can  get ourselves a deep learning model, but basically we're gonna have to train one.  So, I've got a couple of examples: I've got a Kaggle example and a Colab example. Maybe I'll  quickly show you both.
They're gonna be the same thing. And I'm just going to create a dog or a  cat classifier. Okay, so here's our Kaggle model, I'll click on edit so you can actually  see what it looks like in edit view. Now, Kaggle already has fastai installed but I  always put this first, just to make sure we've got the latest version.
And obviously import stuff,  we… So we're going to grab the pets data set, a function to check whether it's a cat, that's  our labeling function for our ImageDataLoaders — remember this is just another way of doing  DataBlocks, it's like a little shorthand. And we create our learner, and we fine tune it.  Okay, so, that's all stuff we've seen before. So, in Kaggle every notebook has an edit-view  (which is what you just saw) and a reader-view.
And so you can share your notebook — if  you want to. And then anybody can read the reader-view, as you see. And so, here, you can  see, it shows you what happened when I ran it. And so, I trained it, it took… you know, the GPUs  on Kaggle are a bit slower than most modern GPUs but they're still fast enough, I mean, it takes  five minutes.
And there's one bit at the end here, which you haven't seen before, which is  I got learn.export and I give it a name. Now, that's going to create a  file containing our trained model, and… that's the only thing! Creating this file  is the only thing you need a GPU for, right? So you do that on Kaggle, or on Colab.
So here's  exactly the same thing on Colab… you can see “pip install”, is_cat, our data, ImageDataLoaders, so  I've got a show_batch here as well, just for fun. Create my learner and then export. So, while  we wait I might go ahead and just run that. One nice thing about Kaggle is once you've run  it and saved it, you can then go to the data tab and, here is basically anything you've  saved, it's going to appear here, and here it is: model.
pickle, right? So now I can  go ahead and download that and that will then be downloaded to my downloads folder. And then I  need to copy it into the same directory that my HuggingFace Spaces App is in. Now my HuggingFace  Spaces App is currently open in my terminal, on Mac you can type “open .” or in Windows you  can type “explorer.exe .” and that'll bring up your finder or explorer in that directory.
  And so then you can just paste that, that thing you downloaded into this  directory. Something by the way, in Windows I do, which I find really helpful is,  I actually grab my… my home directory in Linux and I pin it to my Quick Access and that way I can  always jump in Windows straight to my Linux files. Not really something you have to worry about  on Mac because it's all kind of integrated, but on Windows they're like, kind  of like, two separate machines.
Okay, so… Let's do… so, I created a Space called testing and  I downloaded my “model.pkl” and I pasted it into testing. So now we need to know how  do we do predictions on a saved model? So we've got a notebook for that…  Okay, so we've got a notebook for that and… so I'm going to take you through how we use  a model that we've trained to make predictions.
There's a few funny things with hash pipe  which I'll explain in a moment, just ignore those for now. So we import fastai as usual. We  import Gradio as we did before. And we copy in the exact same is_cat definition we had before.  That's important, any external functions that you used in your labeling, need to be included  here as well because that learner refers t
o those functions, okay? It saves... that  learner saves everything about your model, but it doesn't have the source code of the  function, so you need to keep those with you. So, let's try running this. So, for example, I just grabbed… (as you might have seen in my  explorer) I just popped a dog picture there. And so we can create a python image library image  from that dog, turn it into a slightly smaller one so it doesn't overwhelm my whole screen and  there is a picture of a dog.
So, how do we make predictions of whether that's a dog or a cat?  Well, it's very simple, all we do is… instead of training a learner, we use load_learner(). We pass  in the file name that we saved, and that returns a learner. This learner is exactly the same as  the learner you get when you finish training. So here we are, here's Colab, right? We've just been  training a learner, so at the end of that, there's a learner that's been trained, and so we… We kind  of froze it in time, something called a pickle file — which is a python concept. It's like a  frozen object, we saved it to disk, we transferred
it to our computer, and we've now loaded it,  and we've now unfreezed thought it. This is our unpickled learner and we can now do whatever  we like with that. So one of the things that… One of the methods that a learner has is a dot  predict method, so, if I run it, you can see, even on my laptop, it's basically ~(instance)  instant. In fact we can see how long it took.
If you — in Jupyter things that start with percent  are called magics, they're special Jupyter things, so for example, there's a thing  to see how long something takes. There you go, okay, so it took 54 milliseconds  to figure out that this is not a cat. So it's returning two things: is it a cat? (as  a string); is it a cat? (as a zero or one), and then the probability that it's a dog, and the  probability that it's a cat.
So the probability of zero (false) and one (true) of is  it a cat? so definitely a dog. So we now want to create a Gradio interface  which basically has this information. So Gradio requires us to give it a  function that it's going to call, so here's our function.
So we're going to call  predict, and that returns as we said three things: the prediction is a string; the index of  that; and the probabilities of whether it's a dog or a cat. And what Gradio wants, is it wants  to get back a dictionary containing each of the possible categories — which in this case is dog  or cat — and the probability of each one. So, if you haven't done much python before, a dict  of a zip may be something you haven't seen.
Very handy little idiom well worth checking out.  Ditto if you haven't seen map before, anyway, here it is. One slightly annoying thing about  Gradio at the moment is that it doesn't handle Pytorch Tensors, you can see here Pytorch is  not returning normal numbers, it's returning Tensors. It's not even returning Numpy Arrays.
  In fact radio can't handle Numpy either, so we have to change everything just to a  normal float, so that's all that this is doing, it's changing each one to a float. So, for  example, if I now call classify_image() with our doggy image, we get back a dictionary  of a dog, yes definitely; cat, definitely not. So now we've got all that we can go  ahead and create a Gradio interface.
So Gradio interface is something where  we say, well, what function do you call to get the output, what is the input — in this  case we say, oh, the input is an image — and so check out the Gradio docs, it can be all  kinds of things like a webcam picture, or a text, or you know, all kinds of things.
  Give it a shape that it's going to put it into, the output's just going to be a label so we're  going to create very, very simple interface. And we can also provide some examples, and so  there's a dog, a cat, and a dunno — which I'll tell you about in a moment — which you'll see  here there's a dog, and a cat, and a don't know. So once I launch it, it says, okay, that's  now running on this url.
So if I open that up; you can see now, we have just like Suvash,  we have our own — not yet in production, but running on our own box — classifier.  So let's check: dog, so you can click it and upload one, or  just choose the examples… Here they are! So it's running on  my own laptop, basically instant. And I really have to tell you the story  about this guy here, this is the dunno.
Submit, wait, why is it saying 100? Normally this  says like 50 50. That's a bummer, this model's got… messed up my whole story. So last time I  trained this model, and I ran it on the… dunno, it said, it said like it's almost exactly  50 50. And the way we found this picture is… I showed my six-year-old daughter, she's like:  “what are you doing dad?”, like, I'm coding, “what are you coding?” Oh, you know, dog cat classifier.
  She checks it out and her first question is: “can I take your keyboard for a moment”, and  she goes to google, and she's like: what is a dog mixed with a cat called. Like, there's no such  thing as a dog mixed with a cat. Anyway, she goes to the images tab, and finds this picture, and  she's like: “look there's a dog mixed with a cat”. She said: “run it on that dad!, run it on that!”  And I ran it and it was like: 50-50.
It had no idea if it's a dog or a cat. Now, this model I  just retrained today, now it's sure it's a cat. So there you go, I think I used a slightly  different training schedule or something, or gave it an extra epoch, anyway, so that's a dog-cat,  but apparently it's a cat. I guess it is a cat. It's probably right. I shouldn't  have trained it for as long.
Okay, so there's our interface.  Now, that's actually running, so you actually have to click the  start button to stop it running so otherwise you won't be able to do  anything else in your notebook. Okay, so, now we have to turn that into a python  script. So, one way to turn it into a python script would be to copy and paste into a  python script all the things that you need.
I need to copy and paste into a python  script all these parts of this that you need, so, for example, we wouldn't need this — it's  just to check something out, we wouldn't need this — it was just experimenting. This was  just experimenting. We'd need this, right? So, what I did… is I went through and I wrote  “hash pipe export” at the top of each cell that contains information that I'm  going to need in my final script, and then — so they're the steps, right? And then,  at the very bottom here, I've imported something
called notebook2script from nbdev, and if I run  that, and pass in the name of this notebook, that creates a file for me called app.py  containing that script. So this is a nice easy way to — like when you're working with stuff  that's expecting a script and not a notebook, like HuggingFace Spaces does.
It's fine to just  copy and paste into a text file if you like, but I really like this way of doing it because  that way I can do all of my experimentation in a notebook, and when I'm done, I just have  a cell at the bottom I just run and export it. How does it know to call it app.py? That's because  there's a special thing at the top default export: default_exp, which says what python file name to  create. So that's just a little trick that I use.
So now we've got an app.py. We need to upload  this to Gradio, how do we do that? You just… you just push it to Git. So… you can either do it with  Visual Studio Code, or you can type “git commit”. And then “git push”. And once you've done that…if we change minimal to testing… I think  this hopefully might still be running my previous model — because I didn't push it,  and that way, we can see our crazy dog-cat.
All right, so here it is, you  can see it running in production. So now this is something that anybody  can — if you set it to public — anybody can go here and check out your model, and so  they can upload it. And so, here's my doggy. Yep, definitely a dog.
Cat, yeah I think I might  have trained this for an epoch or two less, so it's less confident. Yeah  definitely a cat. Dog-cat… hey dog-cat… hmm, still thinks it's  definitely a cat. Oh well, so weird. Okay, so, that is…Okay, so, that  is an example of getting a simple model in production. There's a couple of  questions from the forum, from the community. Okay so one person's asking: “what's the difference between a pytorch model and  a fastai learner?” Okay, that's fine, we'll get to that shortly. Don't know if it'd be the lesson,  it might be this lesson or the next lesson.
And then somebody else asked, basically it's  asking: “how many epochs do we train for?” So, as you train a model your error  rate, as you can see, it improves. And so the question is: should I run more?  Should I increase the number of epochs? This is doing three epochs, right? Here's  my three epochs plus one to get started.
Look, it's up to you, right? I mean, for this,  is here saying: there's a one percent error. I'm… I'm okay with the one percent error,  you know, if you want it to be better, then you could use more data augmentation, and you  could train it for longer. If you train for long enough, as we'll learn about soon, and then… maybe  the next lesson.
If you train for long enough, your error rate actually starts getting  worse. And, you'll see, we'll learn about why. So basically yeah, you can train until it's  good enough or until you've run out of patience, or time, or run out of compute, or until you…  or until the error rate starts getting worse. Okay, oh, and then in Colab,  how do you grab your model? All you need to do in Colab  is, after you've exported it, is, if you go into their file browser you'll actually see it here, right? And  you can click download. It's a bit weird,
it doesn't like, pop up a box saying  where do you want to download it to, but instead this kind of progress circle thing  pops up, and so depending on how big it is, and so forth, it can take a few minutes.  And once that circle fills up then it'll, the browser thing will finally pop up and say,  okay, you can save it.
Okay so that's how you… that's how you actually grab your model. So as you  can see, the step where you actually need a GPU, you can use these totally free resources:  Colab, Kaggle, and there are other ones we'll talk about in future lessons. And then you can do  everything else on your own computer, including the predictions, the predictions are fast,  right? So, you really don't need to use a GPU for that, unless you're doing thousands of them.  Okay, here it goes, now it's asking me to save it.
Okay, so now one big issue is, we needed…  to run it on our computer… we needed python and Jupyter Notebooks running  on our computer, so how do you do that? Because this is where often people get in all  kinds of trouble… “I'm trying to figure out how to get this all working.
” So, the good news is  we've actually got something that makes it very, very straightforward. It's called fastsetup.  There's really only just one part of it you need, so let me show you…. it's… it's… actually a  Git repository on Github. Github's the place where most Git repositories live. So if you  go to Github fastai/fastsetup you'll see it. And so what you can do is you can now grab this  whole repository, just by clicking here on code, and if you've got Github Desktop installed,  click on open with Github Desktop and, as you'll see, it brings this up saying: okay  I'm ready to save this for you, so I click clone,
so it's making a copy of it. There we go… So, basically, once you've cloned it, you'll then find there's a file in there  called setup-conda.sh… which, you know, the details don't really matter, it's pretty short,  but, that's the thing that's going to install python for you. So at that point you can just run  “./setup-conda.sh” and it'll run this installer.
Now, if you've got Linux or Mac you've already got  python on your machine. Don't use that python!! And the reason is because that python, it's called  the system python, it's used by your computer to do computery stuff, right? It's actually… it's  actually needed. You don't want to be messing with it, I promise you, like… it's, it always  leads to disaster, always! You want your own development version of python, it's also going  to make sure you've got the latest version, and all the libraries you want, and, by far, the  best one for you is almost certainly going to be
these conda based python distributions, so if  you run setup conda you'll get the one that we recommend. And, the one we recommend at the moment  is something called mambaforge. So basically once you run it, you'll find that you've now —  you close your terminal and reopen it — you'll find you've now got one extra command, which is  called mamba, and mamba lets you install stuff.
So once you've run it, you'll be  able to go “mamba install fastai”. And that's gonna… actually we should probably,  I should mention this, actually more, a bit more detail about how to install it  correctly. If we go to docs.fast.ai, installing. Yeah, okay, we actually want to do “conda install  -c fastchan fastai”. So let's just copy and paste.
Oh sorry, not actually… and then, the other thing  I'll say is, instead of using conda, replace conda with mamba because nowadays it's much  faster. So “mamba install -c fastchan fastai”. Now, this is going to install everything  you need. It's going to install Pytorch, it's going to install Numpy,  it's going to install fastai… and so forth, right? And… so  obviously I've already got it.
And then the other thing you'll  want to do is install nbdev, so you can do exactly the same thing for  nbdev. You don't have to, right? It's just that… but that'll install Jupyter  for you, amongst other things. And so at that point you can now, you can  now use Jupyter. And so the way Jupyter works is — you can see it over here. This is my… (I'll  go ahead and close it so we can start again.
) So basically, to use Jupyter you just type  “jupyter notebook”, okay? And when you run it, it'll say: okay, we're now running a server  for you, and so if you click on that hyperlink, it'll pop up this, okay?, which is exactly what  you see me use all the time. Okay so, you know, that hopefully is enough to kind of get you  started with python, and with Jupyter Notebook… The other way people tend to install software  is using something called pip instead of mamba.
Pretty much anything you can do with mamba you  can also do with pip, but if you've got a GPU, pip isn't going to install things generally so  that it works on your GPU, you have to install lots of other stuff, which is annoying, so,  that's why I kind of tell people to use mamba, but you can use pip otherwise. Still a little bit of  red. Please let us know how we can help you again.
Okay so let's see how we’re going with  our steps. I forgot I had these steps here to remind myself. We created a space,  tick. We created a basic interface, tick. Okay we got git setup, we got conda set up, or  mamba. So mamba and conda are the same thing, mamba is just a much faster version and we'll keep  some notes on the course website because at the moment they're actually working on including the  speed ups from mamba into conda.
So at some point maybe it'll be fine to use conda again. At the  moment conda is way too slow, so don't use it. Okay we've done dogs versus cats no  problem. Yeah so we could also look at pet breeds (well yeah we'll briefly look at  that.) Okay, we've used an exported learner, no problem. We’ve used nbdev, no problem. Oh!  okay… try the api… alright this is interesting.
So I think we can all agree, hopefully, that  this is pretty cool that we can provide to anybody who wants to use it, for free, a  real working model, and you know, with gradio there's actually, you know, a reasonable amount  of flexibility, around like how you can make your website look, you know, using  these various different widgets.
It's not amazingly flexible, but it's  flexible enough to kind of… it's really just for prototyping, right. So gradio has a  lots of widgets and things that you can use. The other main platform at the moment that  HuggingFace Spaces supports is called streamlit. streamlit is more flexible, I would say, than  gradio - not quite as easy to get started with but you know, it's kind of that nice in  between, I guess, so also a very good thing you know again mainly for, kind of, building  prototypes. But at some point you're going to
want to build more than a prototype. You want  to build an app, right, and one of the things I really like about gradio in HuggingFace  Spaces is… there's a button down here “view the api”. So we can actually create any  app we want and the key point is that the thing that does the actual model predictions for us is  going to be handled by HuggingFace Spaces Gradio, right, and then we can write a Javascript  application that then talks to that.
Now there's going to be two reactions here …  Anybody who's done some front-end engineering is going to be like, “oh great, I can now  literally create anything in the world, because I just write any code and I can do it and  they'll be excited.” And a lot of data scientists might be going “uh oh… I have no idea how to use  javascript, it's not in my, you know, inventory.
” So this is again where I'm going to say:  look, don't be too afraid of Javascript. I mean obviously one option here is just to kind of  say, “hey I've got a model” and throw it over to the wall to your mate who does know Javascript and  say please create a Javascript interface for me. But let me just give you a sense of  how really not hard this actually is.
So there's a… end point. There's now a  url that's running with our model on it and if you pass it some data… an image…  some image data… to this… to this url, it's going to return back the dictionary.  So it's going to do exactly the same thing that this UI does but as an  api, as a function we can call.
And so it's got like, examples here of how  to call it… so for example, i can actually… let me show you the api as an example using  that minimal interface we had … because it's just going to be a bit simpler. So if I click  curl and I copy that… copy that… and paste. So you can see there … (oh that's not a great  example passing in hello world) …if I pass in Tanishq again, let's see how  I'm going with his name Tanishq.
Here you can see it returns back “Hello  Tanishq!” So this is how these apis work, right. So we can use javascript to call  the api and we've got some examples. So I've created a website, and  here is my website “tinypets” and on this website as you can see it's not  the most amazingly beautiful thing but it's a website.
It's a start right, and up here I've  got some examples, here we go, single file, click, choose file, click… and in this example I'm  actually doing full pet classification. So I actually trained a model to classify breed,  which we'll talk about more next week, rather than just dog versus cats. So let's  pick a particular breed… and we run it… Oh! And there it is! Now not very amazing right,  but the fact is that this is now a Javascript App means we have no restrictions about what we  can do.
And let's take a look at that HTML… that's it! It easily fits in a  screen, right, and the basic steps are not crazy, right, it's basically…  we create an import for our photo, we add an event listener that says when you  change the photo call the read function, the read function says create a file reader, read the  file and when you finished loading, call loaded, and then loaded says… fetch  that… now that path there… is… that path there! Right! Except  we're doing the full pets one.
So this is basically just copied and  pasted from their… from their sample. And then grab the JSON, and then grab from the  data the first thing… the confidences… the label, and then set the HTML. So as you can see it's like  (okay) if you haven't used javascript before these are all new things, right, but they're not… it's  not harder than python, right, it's just… it's just another just another language to learn.
And  so from here you can start to build up, right! So for example we've created a multi-file version. So with the multi-file  version, let me show you… multi-file… choose… so we can now click a few… so we've got a newfie,  a ragdoll, a basset hound, and some kind of cat. I'm not much of a cat person.  So we chose four files and bang they've all been classified! Apparently it's a  Bengal! (I wouldn't know) Here's our newfoundland.
So there's the multifile version, and if you  look at the code, it's not much more, right, it's now just doing… it's getting all the files  and mapping them to read, and now appending each one, so not much more code at all. And  as you might have seen on our site here, there's a few more examples, which is  some of the community during the week, has created their own versions.
So this one  here is… I think this is… the yeah… this is from one of the gradio guys... they called it  “get to know your pet.” So if i choose a pet… I kind of… I really like this because it actually  combines two models. So first of all it says, oh! It's a basset hound, and then it lets me type  in and ask things about it. So i can see, “oh, w
hat kind of tail does it have?...” search,  and so that's now going to call an NLP model, which asks about this… it's a curved saber tail.  There we go! “What maintenance does it need?” So again, like here you can kind of see how (oh) a  basset hound’s ears must be cleaned inside and out frequently. So this is like combining models.So  you can see this is something that you couldn't do with just a kind of a… a ready to go interface.
And so the next thing i wanted to point out is how did we create the website? I showed you  how to create an HTML file, but like how do you create those, and how do you make a website out  of them? Well watch this! Let's… here's…. here's…. here's the source code to our most basic version,  okay! So i could just save this… there we go… Okay so we could open that with Visual Studio  Code… and… what we could actually do is, we could… here it just is in explorer or mac  & finder. I could just double click on it
and here it is! It's a working app!! So you can see i don't need any software  installed on my computer to use a Javascript App, right! It's a single file… I just run it  in a browser… a browser is our complete execution environment. It's a got a  debugger. It's got the whole thing. So here you can (yeah) here you can  see it's… it's just calling out to this external hacking faces endpoint, so I  can do it all sitting here on my computer.
So once I've got my HTML file that's  working fine on my computer, in VSCode, how do I then put it on the web,  so that other people can use it? Again, the whole thing's free. There's a really  cool thing called github pages, which basically will host your website for you.
And because it's  just Javascript, it'll… it'll all work just fine! The easiest way to create a  github pages site in my opinion, is to use something called fastpages,  which is a fast.ai thing! And… basically all you do… is you  follow the setup process. So first it does it's… let's just go  through it… so it says generate a copy by clicking on this link.
So I click the  link, all right, okay, give it a name… (I try to make everything public! I  always think it's good good practice… you don't have to) …create repo, generating, okay, and then there's basically two  more steps, takes about five minutes. We don't have five minutes, so I'll show you the one that  I've already built, which is “fast.ai/tinypets” And so once it's done you'll basically end up with  this empty site which, again, you just go “code” “open with github desktop” or “open with visual  studio” whatever… so “open with github desktop”...
or you can copy & paste this to your terminal…  and so any one of those is going to get you this whole thing on your computer. You can save your  HTML files there… push it back up to github. And what you'll find is we… we've… FastPages  will show you the link to the website that is created for you.
Now, the website that's created  for you, you can make it look however you want using something called a “Theme”. So you'll see  it's created a file called config.yml where you can pick a theme. So, in this case, I picked a  theme called “alembic” for no particular reason. So Github Pages uses something called “Jekyll”,  and so any “Jekyll” theme will, will basically work.
So I picked out this theme, and so, as a  result, when I now save things into this repo, they will automatically appear in this website,  and the files automatically appear up here in this list. So, if you look at my index, that's  the home page… the entire file is just this, right! The only slightly weird thing  is at the top of every Github Pages file, you have to have three dashes, title,  and layout, and three dashes.
It's called “front meta”. And so, once you do that… and  save it, it will appear in your website. So, something else I did then, I was like, okay, well…  that's all very well that fast.ai has created this website, but I don't really like what it looks  like. I want to create a different version. No worries. You can go to “fast.ai/tinypets”  and click “Fork”.
And, when you click “Fork”, it's going to create your own copy. So I did  that under my personal account which is jph00. And look! Now I've got my own version  of it, and now I can make changes here. So I made a few changes. One change I made was:  I went to config.yml and I changed the theme… to “pages-themes/hacker”. So once you fork, one thing you have to do, which  normally FastPages does for you, is you do have to go to “Settings”, and click “Pages”, and actually  enable “Github Pages”.
So you basically have to, by default, it's turned off, so here you'll just  have to turn it on. So use the “Master Branch Root Save” and then it'll say, “No worries,  it's ready to be published”. And so I changed the config.yml file to point at a different theme.  And so, if you look at now… the jph's Tiny Pets… it's different! Okay? So it's got the same info,  but it's much more “hackerish” because jph00 is a serious hacker, as you  can tell from his website.
So, anyway, look, it's a very brief taste of this  kind of world of Javascript and websites and so forth, but I wanted to give you a sense of, like,  you know, you don't need any money. You don't need any IDEs. You know, you don't really need much  code to get started with writing your own web apps, and, thanks to hugging face spaces,  you know, they'll even host your model for you and all you need to do is just have  the magic string, as a thing to call.
Okay! So… signing out: Hacker Jeremy Howard. Thanks  very much for watching, and, in the next lesson, we're going to be digging into some natural  language processing. We're going to be doing some of the same stuff, but we're going to be  doing it with language rather than pictures. And we're going to be diving under the hood  to see how these models actually work.
We're going to learn about things like stochastic  gradient descent, and we might even be having to brush off a little bit of calculus. I hope  I haven't put you off by saying the “C” word. I will see you next time! Thanks all! Bye!

---

# 3

Hi everybody, and welcome to lesson three of  Practical Deep Learning for Coders. We did a  
quick survey this week to see how people feel that  the course is tracking and over half of you think  
it's about the right pace, and of the rest, some  of you think it's a bit slow and some of you think  
it's a bit… sorry… think it's a bit slow and some  of you think it's a bit fast. So, hopefully that's   about the best we can do generally speaking. The  first two lessons are a little more easy pacing  
for anybody who's already familiar with the basic  technology pieces, and then the later lessons get,  
you know, more into some of the foundations. Today  we're going to be talking about things like matrix  
multiplications and gradients and calculus  and stuff like that. So, for those of you  
who are more mathy and less computery you might  find this one more comfortable, and vice versa.
So remember that there is an official  course updates thread where you can see  
all the up-to-date info about everything you  need to know, and of course the course website  
as well. So by the time, you know, you watch  the video of the lesson it's pretty likely that,  
if you come across a question, or an issue,  somebody else will have, so definitely search  
the forum and check the facts first and then, of  course, feel free to ask a question yourself on  
the forum if you can't find your answer. One  thing I did want to point out which you'll  
"Lesson 0" How to fast.ai
see in the lesson thread and the course website  is there is also a Lesson Zero. Lesson Zero is  
based heavily on Radek’s book Meta Learning, which  internally is based heavily on all the things  
that I've said over the years about how to learn  fastai. It's… we try to make the course full of  
titbits about the science of learning itself  and put them into the course. It's a different  
course to probably any other you've taken and  it's, I strongly recommend watching Lesson Zero,  
as well. The last bit of Lesson Zero is about how  to set up a Linux box from scratch — which you can  
happily skip over unless that’s of interest — but  the rest of it is full of juicy information that I  
think you'll find useful. So the basic  idea of what to do, to do a fast.ai lesson  
How to do a fastai lesson
is: watch the lecture and I generally, you  know, on the video, recommend watching it  
all the way through without stopping once and  then go back and watch it with lots of pauses,  
running the notebook as you go. Because otherwise  you're kind of like running the notebook   without really knowing where it's heading, if that  makes sense. And the idea of writing the notebook  
is, you know, there's a few notebooks you could go  through, so obviously there's the book! So going  
through chapter one of the book, going through  chapter two of the book as notebooks, running   every code cell and experimenting with inputs and  outputs to try and understand what's going on.  
And then trying to reproduce those results,  and then trying to repeat the whole thing  
with a different data set. And if you  can do that last step, you know, that's  
quite a stretch goal, particularly at the start of  the course, because there's so many new concepts   but that really shows that you… you've got it  sorted. Now, this third bit: “reproduce results”,  
I recommend using… you'll find in the fastbook  repo — so the repository for the book — there  
is a special folder called “clean”, and “clean”  contains all of the same chapters of the book  
but with all of the text removed — except  for headings — and all the outputs removed,  
and this is a great way for you to  test your understanding of the chapter,  
is… before you run each cell, try to  say to yourself, “okay, what's this for  
and what's it going to output? — if anything.”  And if you, kind of, work through that slowly…  
that's a great way… and any time you're not  sure, you can jump back to the version of   the notebook with the text to remind yourself  and then head back over to the clean version.
So, there's an idea for something which a lot of  people find really useful for self-study. I say  
How to not self-study
self-study but, of course, as we've mentioned  before, the best kind of study is study done,  
to some extent, with others — for most people. You  know, the research shows that you're more likely  
to stick with things if you're doing it as kind of  a bit of a social activity. There are… the forums  
are a great place to find and create study groups,  and you'll also find on the forums a link to  
our Discord server. So… yes, our Discord server  where there are some study groups there as  
well. So, you know, in person study groups,  virtual study groups, are a great way to,  
you know, really make good progress and  find other people at a similar level to you.  
If there's not a study group going at your level,  in your area, in your time zone; create one,  
so just post something saying: Hey! Let's create  a study group. So this week there's been a lot  
of fantastic activity. I can't show all of it so,  what I did was, I used the summary functionality  
Highest voted student work
in the forums to grab all of the things with the  highest votes and so, I just quickly show a few   of those. We have a Marvel Detector created this  week. Identify your favorite marvel character.  
I love this: a rock, paper, scissors game where  you actually use pictures of the rock, paper,   scissors symbols — and apparently the computer  always loses, that's my favorite kind of game.  
There is a lot of Elon around, so very handy to  have an Elon Detector to, you know, either find  
more of him if that's what you need, or maybe less  of him. I thought this one was very interesting.  
I love these kind of really interesting ideas  there's like… “gee I wonder if this would work?”   Can you predict the average temperature  of an area based on an aerial photograph?  
And the… and apparently the answer is  yeah! Actually you can predict it pretty   well. Here in Brisbane, it was predicted,  I believe to within one and a half celsius.  
I think this student is actually a genuine  meteorologist, if I remember correctly – he   built a cloud detector. So, then, building on  top of the What's Your Favorite Marvel Character,  
there's now… also this is a Marvel  character. My daughter loves this one:   what dinosaur is this? And, I'm not as good about  dinosaurs as I should be. I feel like there's  
10 times more dinosaurs than there was when  I was a kid, so I never know their names,   so this is very handy. This is cool, choose your  own adventure, where you choose your path using  
facial expressions. And, I think this music  genre classification is also really cool.  
Brian Smith created a Microsoft Power App  application that actually runs on a mobile phone.  
That's pretty cool. Wouldn't be surprised to  hear that Brian actually works at Microsoft, so,   also an opportunity to promote his own stuff  there. I thought this art movement classifier  
was interesting in that, like, there's a really  interesting discussion on the forum about   what it actually shows, about similarities  between different art movements.  
And I thought this redaction detector project was  really… was really cool as well – and there's a  
whole tweet thread and blog post and everything  about this one particularly great piece of work.
Okay so I'm going to quickly show  you a couple of little tips before  
Pets breeds detector
we kind of jump into the mechanics  of what's behind a neural network,   which is, I was playing a little bit with how do  you make your neural network more accurate during  
the week and so I created this Pet Detector.  And this pet detector is not just predicting…  
predicting dogs or cats but what breed is it.  That's obviously a much more difficult exercise.  
Now, because I put this out on HuggingFace Spaces,  you can download and look at my code, because,  
if you just click files and versions on the space  — which you can find a link on the forum and the   course website — you can see them all here  and you can download it to your own computer.
So I'll show you what I've got here.  Now one thing I'll mention is today  
Paperspace
I'm using a different platform, so in the past  I've shown you Colab and I've shown you Kaggle  
and we've also looked at doing stuff on your own  computer — not so much training models on your   computer, but using the models you've trained  to create applications. Paperspace is a another  
website — a bit like Kaggle and Google — but in  particular they have a product called Gradient  
Notebooks, which is — at least as I speak (and  things change all the time, so check the course  
website) — but, as I speak, in my opinion, is  by far the best platform for running this course  
and for, you know, doing experimentation. I'll  explain why as we go. So, why haven't I been  
using the past two weeks? Because I've been  waiting for them to build some stuff for us to  
make it particularly good and they just… they just  finished. So, I've been using it all week and it's   totally amazing. This is what it looks like…  so you've got a machine running in the cloud,  
but the thing that is very special about it is  it's a… it's a real… it's a real computer you're  
using. It's not like that kind of weird virtual  version of things that Kaggle or Colab has,  
JupyterLab
so if you whack on this button down here  you'll get a full version of Jupyter Lab,  
or you can switch over to a full version of  classic Jupyter Notebooks. And, I'm actually  
going to do stuff in Jupyter Lab today because  it's a pretty good environment for beginners  
who are not familiar with the terminal — which  I know a lot of people in the course are in that   situation — you can do really everything kind of  graphically. There's a file browser, so here you  
can see I've got my pets repo. It's got a Git  repository thing, you can pull and push to Git.  
And then you can also open up a terminal,  create new notebooks and so forth. So what I  
tend to do with this is, I tend to go into a full  screen so it's kind of like its own whole IDE,  
and so you can see I've got here my terminal,  and here's my notebook. They have free  
GPUs and most importantly there's two  good features: one is that you can pay,  
I think it's eight or nine dollars a month  to get better GPUs, and basically as many as,   you know, as many hours as you want.  And they have persistent storage so,  
with Colab — if you've played with it — you  might have noticed it's annoying you have   to muck around with saving things to Google Drive  and stuff. On Kaggle there isn't really a way of,  
kind of, having a persistent environment,  whereas on Paperspace you have, you know,  
whatever you save in your storage it's going  to be there the next time you come… come back. So, I'm going to be adding walkthroughs  of all of this functionality,  
so look at… so if you're interested in really  taking advantage of this, check those out.
Make a better pet detector
Okay so I think the main thing that I  wanted you to take away from Lesson Two  
isn't necessarily all the details of: how do you  use a particular platform to train models and  
deploy them into applications through JavaScript  or online platforms, but the key thing I wanted  
you to understand was the concept. There's really  two pieces: there's the training piece, and at the  
end of the training piece you end up with this  model.pkl file, right? And once you've got that,  
that's now a thing where you feed it inputs and  it spits out outputs based on that model that  
you trained, and then… so you don't need,  you know, because that happens pretty fast,   you generally don't need a GPU once you've got  that trained. And so then there's a separate step,  
which is deploying. So I'll show  you how I trained my pet classifier.  
So you can see I've got two IPython notebooks:  one is “app” which is the one that's going to  
be doing the inference in production; one is the  one where I train the model. So this first bit I'm  
going to skip over because you've seen it before,  I create my ImageDataLoaders, check that my data  
looks okay with show_batch, train a ResNet34  and I get 7% accuracy… so that's pretty good.
Comparison of all (image) models
But check this out, there's a link here   to a notebook I created… (actually most  of the work was done by Ross Wightman)  
…where we can try to improve this  by finding a better architecture.  
There are, I think, at the moment — in the Pytorch  image models libraries — over 500 architectures  
and we'll be learning over the course, you know,  what they are? How they differ? But, you know,  
broadly speaking they're all mathematical  functions, you know, which are basically  
matrix multiplications and these non-linearities  such as RELUs — that we'll talk about today.  
Most of the time those details don't matter, what  we care about is three things: How fast are they?   How much memory do they use? and how accurate are  they? And so, what I've done here with Russ is  
we've grabbed all of the models from Pytorch image  models and you can see all the code we've got  
(there's very, very little  code) to create this plot.
Now… (my screen resolution resolution's a bit,  there we go, let's do that) And so, on this plot,  
on the x-axis we've got seconds per sample:  so how fast is it - so, to the left is better,  
it's faster. And on the right is how accurate  is it, so how accurate was it on ImageNet  
in particular. And so, generally speaking,  you want things that are up towards the top  
and left. Now we've been mainly working with  ResNet and you can see down here, here's ResNet18.  
Now resonant 18 is a particularly small and fast  version for prototyping. We often use ResNet34,  
which is this one here. And you can see this,  kind of like, classic model that's very widely  
used (actually nowadays isn't the state of the  art anymore.) So we can start to look up at these  
Try out new models
ones up here and find out some of these better  models. The ones that seem to be the most accurate  
and fast are these LeVit models, so I tried them  out on my pets and I found that they didn't work  
particularly well. So I thought, okay, let's  try something else out; so next up I tried  
these ConvNeXT models. And this one into  here, it was particularly interesting,  
it's kind of like super high accuracy  — it's the, you know, if you want  
0.001 seconds inference time, it's the most  accurate. So I tried that! So how do we try that?  
All we do is, I can say… so the Pytorch  image models is in the Timm module,  
so at the very start I imported that.  And we can say: list_models and pass in  
a glob, a match. And so this is going to show  all the ConvNeXT models, and here I can find  
the ones that I just saw. And all I need  to do is, when I create the vision learner,   I just put the name of the model in as  a string, okay? So, you'll see, earlier  
this one is not a string, that's because it's  a model that fastai provides, the library.  
Fastai only provides a pretty small number, so  if you install Timm — so you'll need to “pip  
install timm” or “conda install timm” — you'll  get hundreds more and you put that in a string.  
So if I now train that, the time for  these epochs goes from 20 seconds   to 27 seconds, so it is a little bit slower,  but the accuracy goes from 7.2 percent  
down to 5.5 percent. So, you know, that's a pretty  big relative difference... 7.2 divided by 5.5...  
Yeah, it's about a thirty percent improvement.  So, that's pretty fantastic and, you know, it's  
been a few years, honestly, since we've seen  anything really beat ResNet that's widely  
available and usable on regular GPUs, so this is,  this is a big step and so this is a, you know,  
there's a few architectures nowadays that really  are probably better choices a lot of the time,  
and these Conv… so, if you are not sure what  to use, try these ConvNeXT architectures.  
You might wonder what the names are about,  obviously tinys, more large, et cetera,   is how big is the model, so that'll be how much  memory is it going to take up, how fast is it.  
And then these ones here that say in22ft1k,  these ones have been trained on more data,  
so ImageNet, there's two different image data  sets: there's one that's got a thousand categories   of pictures, and there's another one that's  about 22,000 categories of pictures. So this  
is trained on the one with 22,000 categories  of pictures. So these are generally going to be  
more accurate on, kind of, standard photos  of natural objects. Okay, so from there I  
exported my model and that's there, okay? So  now I've trained my model and I'm all done.  
You know, other things you could do obviously  is add more epochs, for example, that image  
augmentation; there's various things you can do,  but, you know, I found this is actually pretty,  
pretty hard to beat this by much. If any of you  find you can do better, I'd love to hear about it.  
Get the categories of a model
So then to turn that into an application, I  just did the same thing that we saw last week,  
which was to load the learner; now this  is something I did want to show you:  
the learner, once we load it and call predict,  spits out a list of 37 numbers. That's because  
there are 37 breeds of dog and cat, so these  are the probability of each of those breeds.  
What order are they in? That's an important  question. The answer is that fastai always  
stores this information about categories —  this is a category, in this case a dog or   cat breed — in something called the vocab  object, and it's inside the data loaders.  
So we can grab those categories, and that's  just a list of strings, just tells us the order.  
So if we now zip together the categories and the  probabilities, we'll get back a dictionary that  
tells you… well… like so. So here's that list of  categories and here's the probability of each one.  
And this was a basset hound, so there you can  see… Yep! almost certainly a basset hound.  
So from there, just like last week, we can go and  create our interface and then… and then launch it  
and there we go. Okay so, what did we just do really?  What is this magic model.pkl file.  
What’s in the model
So, we can take a look at the model.pkl  file – it's an object type called a learner  
and a learner has two main things in it: the  first is the list of pre-processing steps  
that you did to turn your images into  things of the model, and that's basically  
this information here. So it's your data  blocks or your ImageDataLoaders or whatever.  
And then the second thing, most  importantly, is the trained model!   And so, you can actually grab the trained model  by just grabbing the .model attribute — so I'm  
What does model architecture look like
just going to call that m — and then if I type  m, I can look at the model. And so, here it is…  
lots of stuff, so what is this stuff? Well,  we'll learn about it all, over time, but  
basically what you'll find is: it contains lots  of layers, because this is a deep learning model,  
and you can see it's kind of like a  tree, that's because lots of the layers   themselves consist of layers. So there's a  whole layer called the TimmBody, which is  
most of it. And then right at the end  there's a second layer called Sequential,  
and then the TimmBody contains something called  “model”, and it can then contain something called  
“stem”, and something called “stages”.  And then “stages” contain 0, 1, 2, etc.
Parameters of a model
So what is all this stuff? Well, let's take a look  at one of them. So to take a look at one of them,  
there's a really convenient method in Pytorch  called get_submodule where we can pass in a,  
kind of, a dotted string navigating through  this hierarchy. So “0.model.stem.1” goes zero,  
model stem, one. So this is going to return  this LayerNorm2d thing. So what is this layer  
Norm2d thing? Well the key thing is: it's got  some code, it's with the mathematical function  
that we've talked about, and then the other thing  that we learned about is it has parameters. So  
we can list its parameters and look at this:  it's just lots and lots and lots of numbers.   Let's grab another example, we could have a  look at “0.model.stages.0.blocks.1.mlp.fc1”  
and parameters… Another big bunch of numbers.  So, what's going on here? What are these  
numbers and where on earth did they come from?   And how come these numbers can figure out whether  something is a basset hound or not? Okay so…
To answer that question we're going  to have a look at a Kaggle notebook  
Create a general quadratic function
“How does a neural network really work?”   I've got a local version of it here, which I'm  going to take you through. And the basic idea is:  
machine learning models are things that fit  functions to data. So we start out with a very,  
very flexible — in fact an infinitely flexible  — as we've discussed function, a neural network,  
and we get it to do a particular thing,   which is to recognize the patterns  in the data examples we give it.
So let's do a much simpler example than  a neural network, let's do a quadratic.  
So let's create a function f which is 3  x² + 2 x + 1. Okay so it's a quadratic,  
with coefficients three, two and one. So,  we can plot that function f, and give it a  
title — if you haven't seen this before things  between dollar signs is what's called latex,   it's basically how we can create kind of typeset  mathematical equations. Okay, so let's run that.  
And so here you can see the function, here  you can see the title I passed it, and here is   our quadratic. Okay, so what we're  going to do is… we're going to  
imagine that we don't know that's the  true mathematical function we're trying   to find — because it's obviously much simpler  than the function that figures out whether  
an image is a basset hound or not — that  we're just going to start super simple. So   this is the real function and we're going  to try to recreate it from some data.  
Now it's going to be very helpful if we have  an easier way of creating different quadratics,  
so I've defined a kind of a general form of a  quadratic here… with coefficients a, b, and c;  
and at some particular point x, it's going to be a  x squared plus b x plus c. And so let's test that.  
Okay so, that's for x equals 1.5,  that's 3 x squared plus 2 x plus 1,  
which is the quadratic we were… did before.
Now we're going to want to create lots of  different quadratics to test them out and find   which one's best, so this is a somewhat advanced  but very, very helpful feature of python that's  
worth learning if you're not familiar with it,  but it's used in a lot of programming languages,   it's called a partial application of a  function. Basically I want this exact function,  
but I want to fix the values of a, b and c,  to pick a particular quadratic. And the way  
you fix the values of the function is: you  call this thing in python called “partial”,   and you pass in the function, and then you pass in  the values that you want to fix. So for example,  
if I now say make a quadratic 3, 2, 1;   that's going to create a quadratic  equation with coefficients 3, 2, and 1.
And you can see, if I then pass in —  so that's now f — if I pass in 1.5,  
I get the exact same value I did before. Okay, so,  we've now got an ability to create any quadratic  
equation we want by passing in the parameters of  the coefficients of the quadratic — and that gives  
us a function that we can then just call as just  like any normal function, so that only needs one   thing now, which is the value of x, because  the other three a, b, and c are now fixed.
So if we plot that function, we'll get exactly the  same shape, because it's the same coefficients.  
Fit a function by good hands and eyes
Okay, so, now I'm going to show an example of  some data… some data that matches the shape of  
this function. But in real life, data is never  exactly going to match the shape of a function,  
it's going to have some noise. So here's a couple  of functions to add some noise. So you can see,  
I've still got the basic functional form  here, but this data is a bit dotted around it.  
The level to which you look  at how I implemented these   is entirely up to you, it's not like super  necessary, but it's all stuff which, you know,  
the kind of things we use quite a lot so this is  to create normally distributed random numbers…  
this is how we set the seed so that each time  I run this I gotta get the same random numbers.  
This one is actually particularly helpful,  this creates a tensor, so in this case a vector  
that goes from -2 to 2, in equal steps and there's  twenty of them, that's why there's 20 steps  
along here. So then my y-values is just  f(x), with this amount of noise added. Okay,  
so — as I say — the details of that don't matter  too much, the main thing to know is we've got some   random data now. And so, this is… the idea  is now we're going to try to reconstruct the  
original quadratic equation, find the one which  matches this data. So how would we do that?
Well, what we can do is, we can create  a function called plot quadratic that,  
first of all, plots our data, as a scatter  plot, and then it plots a function, which  
is a quadratic – the quadratic we pass in. Now,  there's a very helpful thing for experimenting  
in Jupyter Notebooks, which is the @interact  function. If you add it on top of a function,  
then it gives you these nice little  sliders. So here's an example  
of a quadratic with coefficients 1.5 ,1.5, 1.5.  And it doesn't fit particularly well. So how  
would we try to make this fit better? Well, I  think what I'd do is I take the first slider,  
and I would try moving it to the left  and see if it looks better or worse.   That looks worse to me, I think it needs to  be more curvy, so let's try the other way.
Yeah, that doesn't look bad. Let's do  the same thing for the next slider,   have it this way… no, I think that's  worse, let's try the other way…  
okay, final slider… try this  way, no, it's worse… this way…
So you can see what we can do, we can  basically pick each of the coefficients   one at a time. Try increasing a little bit, see if  that improves it. Try decreasing it a little bit,  
see if that improves it. Find  the direction that improves it,   and then slide it in that direction a  little bit. And then when we're done,  
we can go back to the first one and see if we  can make it any better. Now we've done that!
And actually you can see that's not bad, because  I know the answer's meant to be three, two, one.   So they're pretty close. And,  I wasn't cheating, I promise.  
That's basically what we're gonna do.  That's basically how those parameters   are created. But we, obviously, don't have time,  because the, you know, big fancy models have,  
Loss functions
often, hundreds of millions of parameters, we  don't have time to try a hundred million sliders.   So we need something better. Well the first step  is, we need a better idea of like, when I move it,  
is it getting better or is it getting worse?  So, if you remember back to Arthur Samuel's  
description of machine learning that we learned  about on Chapter One of the book, and in Lesson   One… we need some… something we can measure, which  is a number that tells us how good is our model.  
And if we had that, then as we moved these  sliders, we could check to see whether it's   getting better or worse. So this is called a  loss function. So, there's lots of different  
loss functions you can pick but perhaps the  most simple and common is mean-squared-error,  
which is going to be… so it's going to get in our  predictions, and it's got the actuals. And we're   going to go predictions minus actuals squared  and take the mean. So that's mean squared error.  
So, if I now rerun the exact  same thing I had before,   but this time I'm going to calculate  the loss, “the mse,” between  
the values that we predict, f(x); remember where  “f” is the quadratic we created and the actuals…  
“y”. And this time I'm going to add a  title to our function, which is the loss.
So now let's do this more rigorously. We're  starting at a mean squared error of 11.46,  
so let's try moving this to the left and see if  it gets better… no, worse, so I’ll move it to   the right. All right, somewhere around  there. Okay, now let's try this one.  
Okay, best when I go to the right. Okay,  what about “c”? 3.91. It's getting worse,  
so I keep going (sorry about that). And so  now we can repeat that process, right? So  
we've had each of a, b and c move a little bit.  Let's go back to a, can I get any better than  
3.28? Let's try moving left: yeah, left was a  bit better. And for b, let's try moving left:  
worse. Right was better. And  finally c, move to the right…
Oh, definitely better. There we go. Okay  so, that's a more rigorous approach, it's  
Automate the search of parameters for better loss
still manual, but at least we can, like, we don't  have to rely on us to kind of recognize: does it  
look better or worse? So finally we're going to  automate this. So the key thing we need to know  
is: for each parameter, when we move it up, does  the loss get better? Or when we move it down,  
does the loss get better? One approach would be  to try it, right? We could manually increase the  
parameter a bit and see if the loss improves,  and vice versa. But there's a much faster way,  
and the much faster way is to calculate its  derivative. So if you've forgotten what a  
derivative is, no problem, there's lots of  tutorials out there, you could go to Khan   Academy or something like that, but in short, the  derivative is what I just said. The derivative  
is a function that tells you: if you increase  the input, does the output increase or decrease?  
…and by how much. So that's called the  slope, or the gradient. Now the good news is  
Pytorch can automatically calculate  it for you. So, if you went through  
horrifying months of learning derivative rules  in year 11, and worried you're going to have to   remember them all again, don't worry, you  don't! You don't have to calculate any of  
this yourself, it's all done for you.  Watch this. So the first thing to do,  
is we need a function that takes the coefficients  of the quadratic a, b and c as inputs.  
I'm going to put them all in a list, you'll see  why in a moment, I kind of call them parameters.  
We create a quadratic, passing  in those parameters a, b and c.  
This star on the front is a very,  very common thing in python,   basically it takes these parameters and  spreads them out to turn them into a,  
b and c, and pass each of them to the function. So  we've now got a quadratic with those coefficients  
and then we return the mean squared error  of our predictions against our actuals.  
So this is a function that's going to take the  coefficients of a quadratic and return the loss.
So let's try it. Okay so if we start with a, b and c of  1.5, we get a mean squared error of 11.46.  
It looks a bit weird, it says it's a  tensor, so don't worry about that too much,  
in short, in Pytorch everything is a tensor.  A tensor just means that you don't… it doesn't  
just work with numbers, it also works with lists  or vectors of numbers… that's called a 1d-tensor.  
Rectangles of numbers, so tables of numbers is  called a 2d-tensor. Layers of tables of numbers,  
that's called a 3d-tensor, and so forth.  So in this case this is a single number,  
but it's still a tensor, that means it's just  wrapped up in the Pytorch machinery that allows  
it to do things like calculate derivatives.  But it's still just the number 11.46.
So what I'm going to do is I'm going to create  my parameters a, b and c; and I'm going to put   them all in a single 1d-tensor. Now, 1d-tensor is  also known as a rank-1-tensor. So this is a rank  
1 tensor and it contains the list of numbers  1.5, 1.5, 1.5. And then I'm going to tell  
Pytorch that I want you to calculate  the gradient for these numbers,  
whenever we use them in a calculation. And the  way we do that is we just say: requires_grad_().  
So here is our tensor, it contains 1.5 three times  and it also tells us it's… we flagged it to say,  
please calculate gradients for this particular  tensor, when we use it in calculations.  
So let's now use it in the calculation, we're  going to pass it to that quad_mse, that's the   function we just created that gets the mse — the  mean squared error — for a set of coefficients  
and not surprisingly it's the same number we saw  before: 11.46. Okay, not very exciting, but there  
is one thing that's very exciting, which is added  an extra thing to the end called grad function,  
and this is the thing that tells us that,  if we wanted to, Pytorch knows how to  
create — calculate — the gradients for our inputs.  And to tell Pytorch: yes please, go ahead and do  
that calculation, you call “backward()” on the  result of your loss function. Now when I run  
it nothing happens, or it does look like nothing  happens… But what does happen is: it's just added  
an attribute called grad, which is the gradient  to our inputs “abc”. So, if we run this cell,  
this tells me that if I increase a, the loss will  go down. If I increase b, the loss will go down a  
bit less, you know. If they increase c, the loss  will go down. Now, we want the loss to go down,  
right? So that means we should increase a, b and  c. Well, how much by? Well, given that a is… says,  
if you increase a, even a little bit, the  loss improves a lot, that suggests we're   a long way away from the right answer, so  we should probably increase this one a lot,  
this one the second most, and this one the third  most. Okay so this is saying, when I increase  
this parameter, the loss decreases. So in other  words, we want to adjust our parameters a, b and c  
by the negative of these. We want to increase,  increase, increase. So we can do that  
by saying, okay, let's take our “abc” minus  equals (so that means equals “abc” minus)  
the gradient. But, we're just going to like  decrease it a bit, we don't want to jump too far,  
okay? So just we're just going to go  a small distance. So we're going to,   we're just going to somewhat  arbitrarily pick 0.01.
So that is now going to create a new set of  parameters which are going to be a little bit   bigger than before, because we subtracted negative  numbers. And we can now calculate the loss again.  
So remember, before it was 11.46, so hopefully  it's going to get better. Yes it did,  
10.11. There's one extra line of code, which we  didn't mention, which is with torch.no_grad().  
Remember earlier on, we said that the  parameter abc requires grad, and that   means Pytorch will automatically calculate its  derivative when it's used in a… in a function.  
Here it's being used in a function, but we don't  want the derivative of this, this is not our loss,  
right? This is us updating the gradients. So  this is basically the standard inner part of the  
Pytorch loop, and every neural net deep learning  machine… pretty much every machine learning model,  
at least of this style, that you’ll  build, basically looks like this.   If you look deep inside fastai source code, you'll  see something that basically looks like this.
So we could automate that, right? So let's  just take those steps, which is, we're going to  
calculate — let's go back to here — we're  going to calculate the mean squared error for  
our quadratic. Call backward and then subtract the  gradient times the small number from the gradient.  
So let's do it five times. So, so  far we're up to a loss of 10.1.   So we're going to calculate our loss, call  dot backward to calculate the gradients  
and then: with no_grad, subtract the gradients  times a small number, and print how we go.  
And there we go, the loss keeps improving.  So we now have… some coefficients.  
And there they are: 3.2, 1.9, 2.0. So they're  definitely heading in the right direction.
So that's basically how we do… it's called  optimization. Okay, so you'll hear a lot in  
deep learning about optimizers, this  is the most basic kind of optimizer,   but they're all built on this principle,  of course, it's called gradient descent.  
And you can see why it's called gradient  descent: we calculate the gradients   and then do a descent, which is, we're trying  to decrease the loss. So, believe it or not,  
The mathematical functions
that's the entire foundations of how we create  those parameters. So we need one more piece,  
which is what is the mathematical function  that we're finding parameters for?   We can't just use quadratics, right?  …because it's pretty unlikely that  
the relationship between parameters and  whether a pixel is part of a basset hound  
is a quadratic. It's going to be  something much more complicated.
No problem, it turns out that we can create an  infinitely flexible function from this one tiny  
ReLu: Rectified linear function
thing: this is called a Rectified Linear Unit.  The first piece, I'm sure you will recognize,  
it's a linear function. We've got our  output y, our input x, and coefficients m  
and b. This is even simpler than  our quadratic, and this is a line.  
And torch.clip() is a function that takes that  output, y, and if it's greater than that number,  
it turns it into that number. So in other words  this is going to take anything that's negative,   and make it zero. So this function is going to  show two things: calculate the output of a line,  
and if it is ~(bigger,) ah, smaller  than zero, it'll make it zero.   So that's rectified_linear(). So let's use partial  
to take that function and set the m and b  to one and one. So this is now going to be,   this function here, will be: y equals x  plus one followed by this torch.clip().
And here's the shape, okay. As you'd expect,  it's a line until it gets under zero,  
when it becomes, well, it's still a  line, it's a… becomes a horizontal line.
So we can now do the same thing, we  can take this plot function and make   it interactive using interact. And we can see  what happens when we change its two parameters:  
m and b. So we're now plotting the rectified  linear and fixing m and b. So m is the slope,  
okay, and b is the intercept,  with a shift up and down.  
Okay, so that's how those work. Now, why is this  interesting? Well, it's not interesting of itself,  
Infinitely complex function
but what we could do is, we could  take this rectified linear function  
and create a double ReLU, which adds up  two rectified linear functions together.  
So there's some slope m1, b1 – some second slope  m2, b2 – we're gonna calculate it at some point x.  
And so let's take a look at what that  function looks like if we plot it.  
And you can see what happens is we get  this downward slope, and then a hook,   and then an upward slope. So if I change m1 it's  going to change the slope of that first bit,  
and b1 is going to change its position, okay. And  I'm sure you won't be surprised to hear that m2  
changes the slope of the second bit and b2 changes  that location. Now, this is interesting… why?  
Because we don't just have to do a double ReLU,  we could add as many ReLUs together as we want,  
and if we add as many ReLUs together as we want,  then we can have an arbitrarily squiggly function,  
and with enough ReLUs, we can match it  as close as we want, right? So you could  
imagine incredibly squiggly, like I don't  know, like an audio waveform of me speaking  
and if I gave you 100 million ReLUs to add  together, you could almost exactly match that.
Now, we want functions that are not just… that  we've plotted in 2d, we want things that can  
have more than one input, but you can add these  together across as many dimensions as you like.   And so, exactly the same thing  will give you a ReLU over surfaces,  
or a ReLU over 3d, 4d, 5d and so  forth. And it's the same idea,  
with this incredibly simple foundation you can  construct an arbitrarily, accurate, precise model.
Problem is, you need some numbers for  them, you need parameters. Oh! no problem,  
we know how to get parameters, we use  gradient descent. So believe it or not,  
we have just derived deep  learning! Everything from now on is  
tweaks to make it faster, and make it need  less data, you know, this is… this is it!!  
Now, I remember a few years ago when  I said something like this in a class,   somebody on the forum was like, this reminds  me of that thing about how to draw an owl.  
Jeremy's basically saying, okay, step one: draw  two circles; step two: draw the rest of the owl.  
The thing I'm… I find I have a lot of trouble  explaining to students is: when it comes to   deep learning, there's nothing between these two  steps. When you have ReLUs getting added together,  
and gradient descent to optimize the parameters,  and samples of inputs and outputs that you want,  
the computer draws the owl, right? that's, that's,   that's it, right? So we're going to learn  about all these other tweaks, and they're  
all very important, but when you come down to like  trying to understand something in deep learning,  
just try to keep coming back to remind yourself of  what it's doing, which it's using gradient descent  
to set some parameters to make a wiggly  function which is basically the addition of   lots of rectified linear units or something  very similar to that, match your data.
A chart of all image models compared
Okay, so we've got some questions on the forum.  
Okay, so, the question from Zakia,  with six upvotes… (so for those of you  
watching the video, what we do in the  lesson is we want to make sure that the   questions that you hear answered are  the ones that people really care about  
so, we pick the ones which get the most  upvotes.) This question is: “is there  
perhaps a way to try out all the different models  and automatically find the best performing one?”
Yes, absolutely you can do that, so…
If we go back to our training script,  remember, there's this thing called   list_models() and it's a list of strings,  
so you can easily add a for loop around this that  basically goes, you know, for architecture in  
timm.list_models and you could do the whole lot,  which would be like that, and then you could…  
do that, and a way you go. It's going to take  a long time for 500 and something models,  
so generally speaking, like, I've never done  anything like that myself, I would rather look  
at a picture like this and  say like: Okay, where am I in?   The vast majority of the time — this  is something, this would be the biggest  
error and number one mistake of beginners  I see — is that they jump to these models  
from the start of a new project. At the start  of a new project I pretty much only use ResNet18  
because I want to spend all of my time trying  things out. I'm going to try different data   augmentation. I'm going to try different ways of  cleaning the data. I'm going to try, you know,  
different external data I can bring  in. And so I want to be trying lots of   things and I want to be able to try  it as fast as possible, right? So…
Trying better architectures is the very  last thing that I do. And what I do is,  
once I've spent all this time, and I've  got to the point where I've got… okay,   I've got my ResNet18, well maybe, you  know, ResNet34 because it's nearly as fast.  
And I'm like, okay well, how accurate is it?  How fast is it? Do I need it more accurate,  
for what I'm doing? Do I need it faster, for what  I'm doing? Could I accept some trade-off to make  
it a bit slower, to make it more accurate?  And so then I'll have a look and I'll say:   okay well, I kind of need to be somewhere around  0.001 seconds and so I try a few of these.  
So that would be how I would think about that.
Do I have enough data?
Okay, next question from the forum  is around: “How do I know if I have   enough data? What are some signs that  indicate my problem needs more data?”  
I think it's pretty similar  to the architecture question.   So, you've got some amount of data. Presumably  you've, you know, you've started using all the  
data that you have access to, you've built your  model, you've done your best. Is it good enough?  
Do you have the accuracy that you need for  whatever it is you're doing? You can't know until  
you've trained the model, but as you've seen, it  only takes a few minutes to train a quick model.  
So, my very strong opinion is that the  vast majority of projects I see in industry  
wait far too long before they train their first  model. You know, in my opinion you want to train  
your first model on day one with whatever CSV  files or whatever that you can hack together.  
And you might be surprised that none of the  fancy stuff you're thinking of doing is necessary   because you already have a good enough accuracy  for what you need. Or you might find quite the  
opposite, you might find that, oh my god,  we're basically getting no accuracy at all,  
maybe it's impossible. These are things you  want to know at the start, not at the end.
We'll learn lots of techniques both in  this part of the course and in Part Two   about ways to really get the most out of your  data. In particular there's a reasonably recent  
technique called semi-supervised learning which  actually lets you get dramatically more out of   your data, and we've also started talking already  about data augmentation, which is a classic  
technique you can use. So, generally speaking,  it depends how expensive is it going to be to get   more data. But also what do you mean when you say  “get more data.” Do you mean more labeled data?  
Often it's easy to get lots of inputs  and hard to get lots of outputs.  
For example, in medical imaging — where I spent  a lot of time — it's generally super easy to jump  
into the radiology archive and grab more CT scans.  But it might be very difficult and expensive to,  
you know, draw segmentation masks, and pixel  boundaries and so forth, on them. So often,  
you can get more, you know, in this  case images, or texts, or whatever.  
And maybe it's harder to get labels. And again,  there's a lot of stuff you can do, using things   like we'll discuss: semi-supervised learning to  actually take advantage of unlabeled data as well.
Okay, a final question here: “In the quadratic  example, where we calculated the initial  
Interpret gradients in unit?
derivatives for a, b and c, we got values of  -10.8, -2.4, etc. What unit are these expressed  
in? Why don't we adjust our parameters by  these values themselves?” So I guess the   question here is: “Why are we multiplying it by  a small number?” …which in this case is 0.01.  
Okay let's take those two parts of the question.
What's the unit here? The unit is:  ~(for each increase in “x” of one,  
how much does…) sorry, for each increase in “a”  of one, so if I increase “a” from, in this case,  
we're at 1.5; so if we increase from 1.5 to 2.5,  what would happen to the loss? And the answer is:  
it would go down by 10.9887. Now, that's  not exactly right because it's kind of like…  
it's kind of like, in an infinitely small space,  right? …because actually it's going to be curved,   right? But it's, if it stays…it stayed at  that slope, that's what would happen. So if we  
increased “b” by one, the loss would  decrease — if it stayed constant,  
you know, if the slope stayed the same —  the loss would decrease by minus 2.122.
Learning rate
Okay, so why would we not just change it  directly by these numbers? Well, the reason is…  
the reason is that if we… have  some function that we're fitting,  
and, there's some kind of interesting theory that  says that once you get close enough to the the  
optimal value, all functions look like quadratics  anyway, all right, so we can kind of safely draw  
it in this kind of shape, because this is what  they end up looking like if you get close enough.  
And we're like, let's say we're way out  over here, okay, so we're measuring —  
I use my daughter's favorite pens, the nice  sparkly ones — so, we're measuring the slope here.  
There's a very steep slope, right? So  that seems to suggest we should jump   a really long way. So we jump a really long way,  
and what happened? Well we jumped way too far.  And the reason is that that slope decreased,  
as we moved along, and so that's  generally what's going to happen,   right? Particularly as you approach the optimal,  generally the slope is going to decrease.  
So that's why we multiply the gradient by a small  number. And that small number is a very, very,  
very important number; it has a special  name… It's called the learning rate.
And this is an example of a hyperparameter.  It's not a parameter, it's not one of the  
actual coefficients of your function, but it's  a parameter you use to calculate the parameters.  
Very meta, right? It's a hyperparameter.  And so it's something you have to pick.   Now we haven't picked any yet in any of the stuff  we've done (should I remember) and that's because  
fastai generally picks reasonable defaults —  for most things — but later in the course we  
will learn about how to try and find really good  learning rates, and you will find sometimes you  
need to actually spend some time finding a good  learning rate. You could probably understand the  
intuition here. If you pick a learning rate that's  too big, you'll jump too far and so you'll end up  
way over here, and then you will try to then jump  back again, and you'll jump too far the other way,  
and you'll actually diverge. And so if  you ever see, when your model's training,   it's getting worse and worse, probably  means your learning rate is too big.
What would happen on the other hand if  you pick a learning rate that's too small?  
Then you're going to take tiny steps  and, of course, the flatter it gets  
the smaller the steps are going to get, and  so you're going to get very, very bored. So  
finding the right learning rate is a compromise  between the speed at which you find the answer and  
the possibility that you're actually going  to shoot past it and get worse and worse.
Okay, so one of the bits of feedback I got quite  a lot in the survey is that people want a break  
halfway through, which I think is a good idea.  So I think now is a good time to have a break,   so let's come back in 10 minutes at 25 past seven.
Matrix multiplication
Okay, hope you had a good rest, have a good break,  I should say. So I want to now show you, a really  
really important mathematical computational trick,  which is, we want to do a whole bunch of ReLUs.
All right. So we're going to  be wanting to do a whole lot of   m * x + b. And we want… don't just want… to do m  * x + b. We're going to want to have, like, lots  
of variables. So, for example, every single pixel  of an image would be a separate variable, so we're  
going to multiply every single one of those, times  some coefficient, and then add them all together,  
and then do… the crop… the ReLU. And then  we're going to do it a second time with a  
second bunch of parameters and then a third time  and a fourth time and fifth time. It's going to  
be pretty inconvenient to write out 100 million  ReLUs, but so happens there's a mathematical…  
a single mathematical operation… that does  all of those things for us, except for the   final replace negatives with zeros, and it's  called matrix multiplication. I expect everybody  
at some point did matrix multiplication at high  school. I suspect also a lot of you have forgotten   how it works. When people talk about  linear algebra in deep learning,  
they give the impression you need years of  graduate school study to learn all this linear  
algebra. You don't!! Actually all you need  almost all the time is matrix multiplication  
and it couldn't be simpler. I'm going  to show you a couple of different ways. The first is, there's a really cool  site called “matrixmultiplication.xyz”.  
You can put in any matrix you want, so I'm  going to put in (oops) this one. So this  
matrix is saying I've got three rows of data  with three variables, so maybe they're tiny  
tiny images of three pixels and the value of the  first one is “1 2 1” the second is “0 1 1” and  
the third is “2 3 1” - so those are our three rows  of data. These are our three sets of coefficients,  
so we've got “a b c” in our data so I guess you'd  call it “x1 x2 x3” and then here's our first  
set of coefficients “a b c” “2 6 1” and then our  second set is “5 7 8”. So here's what happens when  
we do matrix multiplication… that second… this  matrix here of coefficients, gets flipped around  
and we do… this is the multiplications and  additions that i mentioned, right! So multiply,  
add, multiply, add, multiply, add. So that's going  to give you the first number because that is the  
left hand column of the second matrix times  the first row, so that gives you the top left  
result. So the next one is going to give us two  results, right! So we've got now the right hand  
one with the top row and the left hand one with  the second row. Keep going down, keep going down,  
and that's it!! That's what matrix multiplication  is - it's multiplying things together and adding   them up. So there'd be one more step to do, to  make this a layer of a neural network, which is  
if this had any negatives we would replace them  with zeros. That's why matrix multiplication  
is “the” critical foundational mathematical  operation in basically all of deep learning.  
So the GPUs that we use… the thing that they are  good at is this, matrix multiplication. They have  
special cores called tensor-cores, which we can  basically only do one thing, which is to multiply  
together two four by four matrices, and then  they do that lots of times for bigger matrices.  
Build a regression model in spreadsheet
So I'm going to show you an example of this.  We're actually going to build a complete  
machine learning model on  real data in the spreadsheet.
So fastai has become kind of famous for a  number of things, and one of them is using  
spreadsheets to create deep learning models.  We haven't done it for a couple of years,   so I'm pretty pumped to show this to you.  What I've done is I went over to Kaggle,  
where there's a competition I actually helped  create many years ago called Titanic, and it's an  
ongoing competition, so 14 000 people have entered  it, or 14000 teams have entered it so far. It's  
just a competition for a bit of fun, there's no  end date, and the data for it, is the data about,  
who… (here it is training data) …who survived  and who didn't, from the real titanic disaster.
And so I… I clicked here on the download button,  to grab it on my computer, that gave me a CSV,  
which I opened up in Excel. The first thing I  did then was I just removed a few columns that  
clearly were not going to be important things,  like the name of the passengers, the passenger id,  
just to try to make it a bit simpler. And so I've  ended up with, each row of this is one passenger.  
The first column is the dependent variable. The  dependent variable is the thing we're trying to   predict. Did they survive? And the remaining  are some information, such as what class of  
the boat - first, second, or third class, their  sex, their age, how many siblings in the family.  
“Par-ch”, I think is parents or something…
So you should always look for a data dictionary,  right? To find out what's… what… number of parents   and children, okay. What was their fare? And  which of the three cities did they embark on,  
Cherbourg, Queenstown, Southampton?  Okay, so there's our data.   Now when I first grabbed it, I noticed that there  were some people with no age. Now there's all  
kinds of things we could do for that, but for  this purpose, I just decided to remove them.  
And I found the same thing for  Embarked. I removed the blanks as well.
But that left me with nearly all  of the data. Okay, so then I've put   that over here. Here's our data  with those rows removed, and…  
okay, that's the… so these… these are the  columns that came directly from Kaggle. So  
basically what we now want to do is we're going  to multiply each of these by a coefficient.  
How do you multiply the  word, Male, by a coefficient?   And how do you multiply “S” by a coefficient? You  can't. So I converted all of these to numbers.  
Male and Female were very easy. I created a column  called isMale, and as you can see there's just an  
IF statement that says, if sex is male then it's  one, otherwise it's zero. And we can do something  
very similar for Embarked. We can have one  column called “did they embark in Southampton?”.  
Same deal, and another column for did  they… what's it called? Cherbourg?
Did they embark in Cherbourg? And now P-class  is one, two, or three, which is a number,  
but it's not really… it's not really a continuous  measurement of something. There isn't one or two  
or three things. They're different levels. So I  decided to turn those into similar things, into  
these binary. These are called “binary categorical  variables”. So, are they first class? and are  
they second class? Okay, so that's all that. The  other thing that I was thinking, well, you know,  
then I kind of tried it and checked out what  happened, and what happened was the people with…  
so I… I created some random numbers. So to create  the random numbers, I just went: equals RAND,  
right? And I copied those to the right, and  then I just went COPY and I went PASTE VALUES.  
So that gave me some random numbers. And that's  my, like… so just because like, I was like…   Before I said: all A, B and C, let's just start  them at 1.5, 1.5, 1.5. What we do in real life  
is we start our parameters at random numbers  that are a bit more or a bit less than zero.  
So these are random numbers. Actually, sorry I  slightly lied, I didn't use RAND, I used RAND  
minus 0.5, and that way I got small  numbers that were on either side of zero.
So, then when I took each of  these, and I multiplied them by  
our fares, and ages, and so forth, what happened  was that these numbers here, are way bigger than,  
you know, these numbers here. And so, in the end  all that mattered was, what was their fare. That’s  
because they were just bigger than everything  else. So I wanted everything to basically go   from zero to one. These numbers were too big. So  what I did up here is, I just grabbed the maximum  
of this column, the maximum of all the fares is  $512. And so then… actually I do age first… I did  
a maximum of age, because similar thing, right?  There's 80 year olds and there's two year olds.  
And so, then over here I just did, okay well,  what's their age divided by the maximum. And so  
that way, all of these are between zero and one.  Just like all of these are between zero and one.  
So that's how I fix… this is  called normalizing the data. Now we haven't done any of these  things when we've done stuff  
with fastai. That's because fastai  does all of these things for you,  
and we'll learn about how, right? But it's all  these things are being done behind the scenes.  
For Fare, I did something a bit more, which is  I noticed there's some lots of very small fares,  
and there's also some… a few very big fares. So  like seventy dollars, and then seven dollars,   seven dollars. Generally speaking when you have  lots of really big numbers, and a few small ones…  
so generally speaking when you've got a few really  big numbers and lots of really small numbers, this   is really common with… with… with money. You know,  because money kind of follows this relationship  
where a few people have lots of it, and they spend  huge amounts of it, and most people don't have   heaps. If you take the LOG of something that's  like… that has that kind of extreme distribution,  
you end up with something that's much more  evenly distributed. So I've added this   here called Log_Fare, as you can see. And these  are all around one, which isn't bad. I could  
have normalized that as well, but I was too  lazy, I didn't bother, because it seemed okay. So at this point you can now  see that if we start from here,  
all of these are, all around the same kind  of level, right? So none of these columns   are going to saturate the others.  So now I've got my coefficients,  
which are, just as I said, they're just random.  Okay, and so now I need to basically calculate  
AX1 plus BX2 plus CX3 plus blah blah blah  blah blah blah blah, okay. And so, to do that,  
you can use SumProduct in Excel. I could have  typed it out by hand, which would be very boring,   but some product is just going to multiply  each of these. This one will be multiplied by…  
where is it… SibSp, by this one. This  one will be multiplied by this one,   so forth and then, they get all added together.
Now one thing, if you're eagle-eyed, you might  be wondering is, in a linear equation we have  
Y equals MX plus B. At the end there's this  constant term. And I do not have any constant  
term. I've got something here called “Const”,  but I don't have any plus at the end. How do we…  
how's that working? Well there's a nice trick that  we pretty much always use in machine learning,  
which is to add a column of data just  containing the number one, every time.   If you have a column of data  containing the number one every time,  
then that parameter becomes your constant term.  So you don't have to have a special constant term,  
and so it makes our code a little bit simpler,  when you do it that way. It's just a trick,  
but everybody does it. Okay, so this is now  the result of our linear model. So this is  
not… I'm not even going to do ReLU, right?  I'm just going to do a plain regression,  
right? Now if you've done regression before, you  might have learned about it as something you kind  
of solve with various matrix things. But in fact  you can solve a regression using gradient descent.
So I've just gone ahead and created a loss for  each row, and so the loss is going to be equal to  
our prediction minus “whether they survived”   squared. So this is going to be our squared  error, and here they all are, our squared errors.  
And so here I've just summed them up. I could  have taken the Mean. I guess that would have   been a bit easier to think about, but SUM is going  to give us the same result. So here's our loss,  
and so now, we need to optimize that  using gradient descent. So Microsoft   Excel has a gradient descent optimizer in  it, called Solver. So I'll click Solver,  
and I just say, okay, what are you  trying to optimize? It's this one here,   and I'm going to do it by  changing these cells here.
And I'm trying to minimize it, and  so we're starting a loss of 55.78.
Actually, let's change it to Mean, as well.  We got Mean or Average? Probably Average.
All right, so start at 1.03.
So optimize that. And there we go. So it's gone from 1.03 to 0.1.  
And so we can check the predictions. So the  first one gets predicted exactly correctly.
It was “they didn't survive”, and we  predicted “they wouldn't survive”.   Ditto for this one. It's very close. And you can  start to see… so this one, you can start to see  
a few issues here, which is like sometimes it's  predicting ~(less than one, sorry) less than zero,   and sometimes it's predicting more than one.  Wouldn't it be cool if we had some way of…  
wouldn't be cool if we had some way of  constraining it to between zero and one,   and that's an example of some of the things we're  going to learn about, that make this stuff work  
a little bit better, right? But you can see it's  doing an okay job. So this is not deep learning,   this is not a neural net, yet. This is just a  regression. So, to make it into a neural net,  
Build a neuralnet by adding two regression models
we need to do it multiple times. So I'm just  going to do it twice. So now rather than  
one set of coefficients, I've got two sets.  And again I just put in random numbers.  
Other than that, all the data's the  same. And so now I'm going to have  
my sum product again. So the first sum  product is with my first set of coefficients,  
and my second sum product is with  my second set of coefficients.   So I'm just calling them linear-one and  linear-two. Now there's no point adding  
those up together, because if you add up two  linear functions together, you get another linear   function. We want to get all those wiggles,  right? So, that's why we have to do our ReLU.  
So in Microsoft Excel, Relu looks like  this: if the number is less than zero,   use zero; otherwise use the number. So that's how  we're going to replace the negatives with zeros.
And then finally, if you remember from our  spreadsheet, we have to add them together.  
So we add the ReLUs together. So  that's going to be our prediction,  
and then our loss is the same as the other  sheet. it's just Survived minus Prediction,   Squared. And let's change that to Mean… Not  Mean… Average. Okay. So let's try solving that.
Optimize $AH$1. And this time  we're changing all of those.  
Solved. So this is using gradient descent. Excel Solver is not the fastest  thing, well, but it gets the job done.  
Okay let's see how we went. 0.08  for our deep learning model versus  
0.1 for our regression. So it's a bit better.  So there you go. So we've now created our first  
deep learning neural network from scratch, and  we did a Microsoft Excel, everybody's favorite  
artificial intelligence tool. So that was a  bit slow and painful. It’ll be a bit faster and  
Matrix multiplication makes training faster
easier if we used matrix multiplication, so let's  finally do that. So this next one is going to be   exactly the same as the last one, but with matrix  multiplication. So all our data looks the same.  
You'll notice the key difference now  is our parameters have been transposed.   So before I had the parameters matching  the data in terms of being in columns.
For matrix multiplication,  the… the expectation is,   the way matrix multiplication works… it works…  is that you have to transpose this. So it goes,  
the X and Y is, kind of, the opposite way around.  The rows and columns are the opposite way around.   Other than that, it's the same. I've  got the same… I just copied and pasted  
the random numbers, so we had exactly the  same starting point and so now… our entire…  
this entire thing here, is a single function  which… which is, matrix multiply, all of this, by  
all of this. And so when i run that,  it fills in exactly the same numbers.
Make this Average. And so now we can optimize that.
Okay, make that a MINIMUM, by changing these.
Solve. You should get the same number, 0.08, wasn’t it?  
Yep, now we do. Okay. So that's just another  way of doing the same thing. So you can see  
that matrix multiplication, it takes like  a surprisingly long time, at least for me,  
to get an intuitive feel for matrix  multiplication, as like a single mathematical  
operation. So i still find it helpful to kind of  remind myself, it’s just doing these sum products,  
and additions. Okay, so that is…  
that is a deep learning neural  network in Microsoft Excel.  
Watch out! it’s chapter 4
And the Titanic Kaggle competition,  by the way, is a pretty fun,  
learning competition. If you haven't done much  machine learning before, then it's certainly   worth trying out, just to kind of get the feel  for these… how these all get put together.
So this is… So the chapter of the book that  this lesson goes with, is Chapter Four.  
And Chapter Four of the book is the chapter  where we lose the most people. Because it's,  
to be honest, it's hard. But part of the reason  it's hard is, I couldn't put this into a book.  
Okay. So we're teaching it in a very different way  in the course, to what's in the book, and you know  
you can use the two together. But if you've tried  to read the book and been a bit disheartened,   yeah, try, you know, try following through…  through the spreadsheet instead. Maybe try  
creating, like, if you use Numbers or Google  Sheets or something, you could try to create   your own kind of version of it on whatever  spreadsheet platform you prefer. Or you could  
try to do it yourself from scratch in Python,  you know, if you want to really test yourself.
So there's some suggestions. Okay. Question from Victor Guerrero. In the  Excel exercise, when Jeremy is doing some  
Create dummy variables of 3 classes
feature engineering, he comes up with two new  columns, Pclass_1 and Pclass_2. That is true.  
Pclass_1 and Pclass_2. Why is there no Pclass_3  column? Is it because Pclass_1… if Pclass_1 is  
zero and Pclass_2 is zero, then Pclass_3 must  be one? So, in a way, two columns are enough to   encode the input with the original column? Yes!  That's exactly the reason. So, there's no need to  
tell the computer about things it can kind of  figure out for itself. So when you create… These   are called dummy variables. So when you create  dummy variables for a categorical variable with  
three levels, like this one, you need  two dummy variables. So, in general,   categorical variable with n levels needs  n-1 columns. Thanks to a good question.
Taste NLP
So what we're going to be doing in our next lesson  is looking at natural language processing. So far  
we've looked at some computer vision, and just  now we've looked at some, what we call, tabular   data, so… so… kind of spreadsheet type data.  Next up we’re… we're going to be looking at  
natural language processing. So I'll give you  a taste of it. So you might want to open up the   Getting Started with… Getting Started  with NLP for Absolute Beginners Notebook.
So here's the Getting Started With  NLP for Absolute Beginners Notebook.   I will say, as a notebook author, I may sound  a bit lame, but I always see when people have  
uploaded it, it always makes me really happy,  so… and it also helps other people find it. So   remember to upvote these notebooks, or any other  notebooks you… you like. I also always read all  
the comments. So if you want to ask any questions  or make any comments, I enjoy those as well.
So natural language processing is about,  rather than taking, for example, image data  
and making predictions, we take text data. That  text data, most of the time, is in the form of  
prose, so like, plain English text. So, you know,  English is the most common language used for NLP,  
but there's NLP models in dozens  of different languages nowadays.
And if you're a non-English speaker,  
you'll find that for many languages, there's less  resources in non-English languages, and there's a  
great opportunity to provide NLP resources in your  language. This has actually been one of the things  
that the fastai community has been fantastic  at, in the global community, is building NLP  
resources. For example, the first  Farsi NLP resource was created  
by a student from the very first fast.ai course.  The Indic languages, some of the best resources  
have come out of fastai alumni and so forth.  So that's a particularly valuable thing you   could look at. So if your language is not well  represented, that's an opportunity, not a problem.
So some examples of things you could use  NLP for? Well perhaps the most common and  
practically useful in my opinion,  is classification. Classification   means you take a document – now when I say a  document that could just be one or two words,  
it could be a book, it could be a Wikipedia  page, so it could be any length. We use the   word “document”, it sounds like that's a specific  kind of length but it can be a very short thing  
or very long thing. We take a document and  we try to figure out a category for it.   Now that can cover many many different kinds of  applications. So, one common one that we'll look  
at a bit is sentiment analysis. So, for example,  is this movie review positive or negative.  
Sentiment analysis is very helpful from things  like marketing and product development - you  
know in big companies there's lots and lots  of, you know, information coming in about your   product. It's very nice to quickly sort it out  and to kind of track metrics from week to week.  
Something like figuring out what author wrote the  document would be an example of a classification  
exercise because you're trying to put in a  category – in this case is, which author. It gives a lot of opportunity in legal discovery.  There's already some products in this area,  
where in this case the category is: is this  legal document in scope or out of scope  
in the court case? Just organizing documents,  triaging inbound emails - so like, which part  
of the organization should it be sent to? Was  it urgent or not? Stuff like that. So these are  
examples of categories of classification.  What you'll find is, when we look at  
fastai NLP library vs Hugging Face library
classification tasks in NLP, is it's  going to look very very similar to  
images. But what we're going to do is  we're going to use a different library. The library we're going to use is called  Hugging Face transformers, rather than fast.ai,  
and there's two reasons for that: the main reason  why, is because i think it's really helpful   to see how things are done in more than one  library; and HuggingFace transformers, yeah so…  
fast.ai has a very layered architecture so you can  do things at a very high level with very little  
code or you can dig deeper and deeper and deeper  getting more and more fine-grained. HuggingFace  
transformers doesn't have the same high level  api, at all, that fast.ai has, so you have to  
do more stuff manually. And so at this point of  the course, you know, we're going to actually  
intentionally use a library which is a little bit  less user-friendly in order to see, kind of, what  
extra steps you have to go through to use other  libraries. Having said that, the reason I picked  
this particular library, is it is particularly  good. It has really good models in it, it has a  
lot of really good techniques in it, not at all  surprising because they have hired lots and lots  
of fast.ai alumni, so they have very high quality  people working on it. So, before the next lesson,  
Homework to prepare you for the next lesson
yeah, if you've got time, take it… take a look  at this notebook and take a look at the data.
The data we're going to be working with is quite  interesting. It's from a Kaggle competition  
which is trying to figure out, in patterns,   whether two concepts are referring to the same  thing or not, where those concepts are represented  
as English text. And when you think about it, that  is a classification task, because the document is,  
you know, basically text one blah, text two blah.  And then the category is similar or not-similar.  
And in fact in this case they actually have  scores, it's either going to be, basically,   zero (0), zero point two five (0.25), point five  (0.5), point seven five (0.75), or one (1), of,  
like, how similar is it. But it's basically a  classification task, when you think of it that   way. So yeah, you can have a look at the data  and, next week, we're going to go through step  
by step through this notebook. And we're going to  take advantage of that, as an opportunity also,   to talk about the really important topics of  validation sets and metrics, which are two  
of the most important topics in, not just deep  learning, but machine learning more generally.
All right, thanks, everybody.  I'll see you next week. Bye.

---

# 4

Hi everybody, and welcome to Practical Deep Learning for Coders Lesson Four, which I think
is the lesson that a lot of the regulars in the community have been most excited about,
because it's where we're gonna get some totally new material — totally new topic, we've
never covered before. We're going to cover natural language processing (NLP), and you'll
find there, there is indeed a chapter about that in the book, but we're going to do it in a totally different way to how it's done in the book. In the book we do NLP using the
fast.ai library, using recurrent neural networks (RNNs).
Today we're going to do something else, which is we're going to do Transformers, and we're
not even going to use the fast.ai library at all in fact. So, what we're going to be
doing today is we're going to be fine-tuning a pre-trained NLP model using a library called
Hugging Face Transformers. Now given this is the fast.ai course, you might be wondering
why we'd be using a different library other than fast.ai.
The reason is that I think that… It's really useful for everybody to have experience and practice of using more than one library.
Because you'll get to see the same concepts applied in different ways, and I think that's
great for your understanding of what these concepts are. Also, I really like the Hugging
Face Transformers library. It's absolutely the state of the art in NLP, and it's well
worth knowing. If you're watching this on video, by the time you're watching it, we will probably have
completed our integration of the Transformers library into fast.ai. So it's in the process of becoming the main NLP (kind of) foundation for fast.ai. So you'll be able to combine
Transformers and fast.ai together. Yeah, so I think there's a lot of benefits to this,
and in the end you're going to know how to do NLP, you know, in a really fantastic library. Now the other thing is, Hugging Face Transformers doesn't have the same layered architecture
that fast.ai has, which means particularly for beginners, the kind of high level, height…
you know… top-tier API that you'll be using most of the time, is not as (kind of) ready
to go for beginners, as you're used to from fast.ai. And so that's actually, I think,
a good thing. You're up to Lesson Four, you know the basic idea now of how gradient descent works, and… and you know, how parameters are learned as part of a flexible function,
I think you're ready to try using a somewhat lower level library that does a little bit
less for you. So it's going to be, you know, a little bit more work. It's still… it's a very well designed library, and it's still reasonably high level, but you're going to
learn to go a little bit deeper. And that's kind of how the rest of the course in general is going to be. On the whole, is, we're going to get a bit deeper, and a bit deeper, and
a bit deeper. Now, so first of all, let's talk about what we're going to be doing with
Finetuning pretrained model
fine-tuning a pre-trained model. We've talked about that in passing before, but we haven't really been able to describe it in any detail, because you haven't had the foundations. Now
you do. You played with these sliders last week, and hopefully you've all actually gone
into this notebook, and dragged them around, and tried to get an intuition for, like this idea of, like, moving them up and down, makes the loss go up and down, and so forth. So
imagine that your job was to move these sliders, to get this as nice as possible, but when
it was given to you, the person who gave it to you said, “Oh! actually slider A, that
should be on 2.0, we know for sure. And slider B, we think it's like around two and a half.
Slider C, we've got no idea.” Now that'd be pretty helpful, wouldn't it, right? Because
you could immediately start focusing on the one we have no idea about, get that in roughly the right spot, and then the one you kind of got a vague idea about, you could just
tune it a little bit, and the one that they said was totally confident, you wouldn't move at all. You would probably tune these sliders really quickly. That's what a pre-trained
model is. A pre-trained model is a bunch of parameters that have already been fitted,
where some of them we’re already pretty confident of what they should be, and some
of them we really have no idea at all. And so fine-tuning is the process of taking those
ones we have no idea what they should be at all, and trying to get them right, and then moving the other ones a little bit.
The idea of fine-tuning a pre-trained NLP model in this way was pioneered by an algorithm
called ULMFiT which was first presented actually in a fast.ai course, I think the very first
ULMFit
fast.ai course. It was later turned into an academic paper by me in conjunction with a
then PhD student named Sebastian Ruder, who is now one of the world's top NLP researchers
and went on to help inspire a huge change, you know, huge kind of step improvement in
NLP capabilities around the world, along with a number of other important innovations at
the time. This is the basic process that ULMFiT described. Step One was to build something
called a language model using basically nearly all of Wikipedia and what the language model
did was it tried to predict the next word of a Wikipedia article. In fact every next
word of every Wikipedia article. Doing that is very difficult. You know, there are Wikipedia
articles which would say things like, you know “the 17th prime number is… dot dot
dot” or “the 40th president of the United States, blah, said at his residence, blah
that”. You know, filling in these kinds of things requires understanding a lot about
how language is structured, and about the world, and about math, and so forth. So to
get good at being a language model a neural network has to get good at a lot of things.
It has to understand how language works at a reasonably good level and it needs to understand
what it's actually talking about, and what is actually true, and what is actually not true, and the different ways in which things are expressed, and so forth. So this was trained
using a very similar approach to what we'll be looking at for fine-tuning but it started with random weights and at the end of it there was a model that could predict more than 30
percent of the time correctly what the next word of a Wikipedia article would be. So in
this particular case, for the ULMFiT paper, we then took that and we were trying to…
the first task I did actually, for the fast.ai course, back when I invented this, was to
try and figure out whether IMDb movie reviews were positive or negative sentiment: Did the person like the movie or not? So what I did was I created a second language model so again
the language model here is something that predicts the next word of a sentence but rather than using Wikipedia I took this pre-trained model that was trained on Wikipedia and I
ran a few more epochs using IMDb movie reviews. So it got very good at predicting the next
word of an IMDb movie review. And then finally I took those weights and I fine-tuned them
for the task of predicting whether or not a movie review was positive or negative sentiment.
So those were the three steps. This is a particularly interesting approach because this very first model, in fact the
first two models, if you think about it they don't require any label. I didn't have to collect any kind of document categories, or do any kind of surveys, or collect anything.
All I needed was the actual text of Wikipedia and movie reviews themselves because the labels
was: “what’s the next word of a sentence?”. Now, since we built ULMFiT, and we used RNNs (recurrent neural networks) for this, at about
the same time-ish that we released this, a new kind of architecture particularly useful for NLP at the time was developed called Transformers. And Transformers were particularly built because
Transformer
they can take really good advantage of modern accelerators like, like Google's TPUs.
They didn't really, kind of, allow you to predict the next word of a sentence. It's
just not how they're structured, for reasons we'll talk about properly in part two of the course. So they threw away the idea of predicting the next word of a sentence and then they,
instead they did something just as good and pretty clever. They took, kind of, chunks of Wikipedia, or whatever text they're looking at and deleted at random a few words and asked
the model to predict which/what were the words that were deleted, essentially. So it's a
pretty similar idea. Other than that the basic concept was the same as ULMFiT. They replaced
our RNN approach with a Transformer model, they replaced our language model approach with what's called a masked language model, but other than that the basic idea was the
same. So today we're going to be looking at models using what's become the, you know,
much more popular approach than ULMFiT which is this Transformers masked language model approach. Okay, John do we have any questions? And I should mention we do have a professor
from University of Queensland, John Williams, joining us, who will be asking the highest
voted questions from the community. What have you got, John? Yeah thanks Jeremy. Look, and we might be jumping the gun here, I suspect this is where
you're going tonight but we've got a good question here on the forum which is: “How do you go from a model that's trained to predict the next word, to a model that can be used
Zeiler & Fergus
for classification”? Sure. So yeah we will be getting into that in more detail and in fact maybe a good place
to start would be the next slide, kind of give you a sense of this. You might remember
in Lesson One we looked at this fantastic Zeiler and Fergus paper where we looked at
visualizations of the first layer of a imagenet classification model and Layer One had sets
of weights that found diagonal edges, and here are some examples of bits of photos that
successfully matched with, and opposite diagonal edges, and kind of color gradients, and here's
some examples of bits of pictures that matched, and then Layer Two combined those and now
you know how those were combined, right, these were rectified linear units that were added together, okay? And then sets of those rectified linear units, the outputs of those, they're
called activations, where then themselves run through a matrix multiply, a rectified linear unit, added together, so that now you don't just have to have edge detectors, but
Layer Two had corner detectors. And here's some examples of some corners that that corner detector successfully found. And remember, these were not engineered in any way, they
just evolved from the gradient descent training process. Layer Two had examples of circle
detectors as it turns out, and skipping a bit, by the time we got to Layer Five we had
bird and lizard eyeball detectors, and dog face detectors, and flower detectors and so
forth. Now, you know, nowadays you'd have something like a resnet50 would be something
you'd probably be training pretty regularly in this course so that, you know, you've got 50-layers, not just 5-layers. Now the later layers do things that are much more specific
to the training task which is, like, actually predicting really, what is it that we're looking
at? The early layers, pretty unlikely you're going to need to change them much, as long
as you're looking at, like, some kind of natural photos, right? You're going to need edge detectors and gradient detectors.
So what we do, in the fine-tuning process, is… there's actually one extra layer after
this, which is the layer that actually says: “What is this?”. You know, it's, it's a dog or a cat or whatever. We actually delete that, we throw it away. So now that last matrix
multiply has one output, or one output per category you're predicting. We throw that
away, so the model now has that last matrix that's spitting out, you know, depends, but
generally a few hundred activations, and what we do is, as we'll learn more shortly in the
coming lesson, we just stick a new random matrix on the end of that. And that's what
we initially train, so it learns to use these kinds of features to predict whatever it is
you're trying to predict. And then we gradually train all of those layers. So that's basically
how it's done and so it's a bit hand wavy but we'll, particularly in part two, actually
build that from scratch ourselves. And in fact in this lesson, time permitting, we're
actually going to start going down the process of actually building a real-world deep neural
net in python, so we'll be starting to actually make some progress towards that goal. Okay
so let's jump into the notebook. So we're going to look at a Kaggle competition that's
US Patent Phrase to Phase Matching Kaggle competition
actually on as I speak, and I created this notebook called “Getting started with NLP
for absolute beginners”. And so the competition is called the “U.S. Patent Phrase to Phrase Matching Competition”.
And, so I'm going to take you through, you know, a complete submission to this competition.
And Kaggle competitions are interesting, particularly the ones that are not playground competitions, but the real competitions with real money applied… they're interesting because this
is an actual project, that an actual organization, is prepared to invest money in getting solved,
using their actual data. So, a lot of people are a bit dismissive of Kaggle competitions
as being, kind of, like, not very real, and it's certainly true you're not worrying about stuff like productionizing the model. But, you know, in terms of like, getting real data
about a real problem that real organizations really care about, and a very direct way to measure the, you know, accuracy of your solution, you can't really get better than this. Okay
so this is a good place, a good competition to experiment with for trying NLP. Now, as
I mentioned here, probably the most widely useful application for NLP is classification
and as we've discussed in computer vision, classification refers to taking an object
NLP Classification
and trying to identify a category that object belongs to. So, previously we've mainly been
looking at images. Today we're going to be looking at documents. Now, in NLP when we
say document, we don't specifically mean, you know, a 20 page long, you know, essay.
A document could be three or four words, or a document could be the entire encyclopedia.
So a document is just an input to an NLP model that contains text. Now, classifying a document,
so deciding what category a document belongs to, is a surprisingly rich thing to do. There's
all kinds of stuff you could do with that. So, for example we've already mentioned sentiment analysis. That's ~(a ca…) a classification task – we try to decide on the category:
positive or negative sentiment. Author identification would be taking a document and trying to find
the category of author. Legal discovery would be taking documents and putting them into
categories according to in- or out-of-scope for a court case. Triaging inbound emails
would be putting them into categories of, you know, throw away, send to customer service,
send to sales, etc. Right? So classification is a very, very rich area, and for people
interested in trying out NLP in real life, I would suggest classification would be the
place I would start, for looking for, kind of, accessible, real world, useful problems
you can solve right away. Now, the Kaggle competition does not immediately look like a classification competition. What
it contains… Let me show you some data…
What it contains is data that looks like this. It has a thing that they call “anchor”, a thing they call “target”, a thing they call “context”, and a score. Now these
are.. I can't remember exact details but I think these are from patents, and I think
on the patents there are various, like, things they have to fill in in the patent, and one
of those things is called “anchor”, one of those things is called “target” and in the competition the goal is to come up with a model that automatically determines
which anchor and target pairs are talking about the same thing. So a score of one here
“wood article” and “wooden article” obviously talking about the same thing. A
score of zero here “abatement” and “forest region” not talking about the same thing. So the basic idea is we're trying to guess the score. And it's kind of a classification
problem, kind of not. We're basically trying to classify things into either “these two things are the same” or “these two things aren't the same”. It's kind of not because
we have not just 1 and 0 but also 0.25, 0.5 and 0.75. There's also a column called “context”,
which is, I believe, is like the category that this patent was filed in and my understanding
is that whether the anchor and the target count as similar or not depends on, you know,
what the patent was filed under. So how would we take this and turn it into something like
a classification problem? So the suggestion I make here is that we could basically say, okay, let's put the, you know,
some constant string like TEXT1 or FIELD1 before the first column and then something
else like TEXT2 before the second column. Oh, and maybe, also the context, I should
have as well TEXT3 in the context, and then try to choose a category of meaning similarity: “Different” “Similar” or “Identical”. So you can basically concatenate those three
pieces together, call that a document and then try to train a model that can predict
these categories. That would be an example of how we can take this, basically, similarity
problem, and turn it into something that looks like a classification problem. And we tend to do this a lot in deep learning, is we kind of take problems that look a bit novel and
different, and turn them into a problem that looks like something we recognize. All right,
so on Kaggle this is a, you know, larger data set that you're going to need a GPU to run.
Kaggle configs, insert python in bash, read competition website
So you can click on the accelerator button and choose GPU to make sure that you're using a GPU. If you click copy and edit on my document I think that will happen for you automatically.
Personally, you know, I like using things like Paperspace generally better than Kaggle,
like, Kaggle's pretty good but you know you only get 30 hours a week of GPU time, and the notebook editor for me is not as good as the real JupyterLab environment. So there's
some information here, I won't go through but it basically describes how you can download
stuff to Paperspace or your own computer as well if you want to. So I basically create
this little boolean, always, in my notebooks called iskaggle which is going to be True if it's running on Kaggle and False otherwise and any little changes I need to make I'd
say “if iskaggle” and put those changes.
So here, you can see here, if I'm not on Kaggle and I don't have the data yet, then download
it. And Kaggle has a little API which is quite handy for doing stuff like downloading data and uploading notebooks and stuff like that, submitting to competitions.
If we are on Kaggle then the data's already going to be there for us which is actually a good reason for beginners to use Kaggle as you don't have to worry about grabbing
the data at all – it's sitting there for you as soon as you open the notebook.
Kaggle has a lot of python packages installed, but not necessarily all the ones you want,
and at the point I wrote this they didn't have the Hugging Faces datasets package, for some reason, so you can always just install stuff. So you might remember the exclamation
mark means this is not a python command, but a shell command, a bash command. But it's
quite neat you can even put bash commands inside python conditionals so that's a pretty
cool little trick in notebooks.
Another cool little trick in notebooks is that if you do use a bash command like “ls”
but you then want to insert the contents of a python variable, just chuck it in parentheses. So, I've got a python variable called “path” and I can go “ls {path}” in parentheses
and that will “ls” the contents of the python variable “path”. So there's another little trick for you.
All right, so when we “ls” that we can see that there's some CSV files. So what I'm
going to do is, kind of, take you through, roughly the process, the kind of process I, you know, went through as, you know… when I first look at a competition. So the first
thing is like, already a data set, indeed, what's in it? Okay, so it's got some CSV files.
You know, as well as looking at it here, the other thing I would do… is I would go to the competition website…
and if you go to “Data”… A lot of people skip over this, which is a terrible idea, because it actually tells you
what the dependent variable means, what the different files are, what the columns are, and so forth. So don't just rely on looking at the data itself but look at the information
that you're given about the data. So, for CSV files, so CSV files are comma separated values, so they're just text files
with a comma between each field, and we can read them using pandas, which for some reason
is always called “pd”. Pandas is one of, I guess, like, (I'm trying to think) probably
Pandas, numpy, matplotlib, & pytorch
like four key libraries that you have to know to do data science in python.
And specifically, those four libraries are: numpy… matplotlib… pandas… and pytorch.
So numpy is what we use for basic, kind of, numerical programming; matplotlib we use for
plotting; pandas we use for tables of data; and pytorch we use for deep learning.
Those are all covered in a fantastic book by the author of pandas which, the new version
is actually available for free, I believe. “Python for data analysis”. So if you're
not familiar with these libraries just read the whole book, it doesn't take too long to
get through, and it's got lots of cool tips and it's very readable. I do find a lot of
people doing this course… often I see people kind of, trying to jump ahead, and… and
want to be like: “Oh I want to know how to, like, create a new architecture” or “Build a speech recognition system” or whatever. But it then turns out that they
don't know how to use these fundamental libraries. So it's always good to be bold and be trying to build things, but do also take the time to, you know, make sure you finish reading
the fast.ai book and read at least Wes McKinney's book. That would be enough to really give
you all the basic knowledge you need, I think. So, with pandas we can read a CSV file and
that creates something called a DataFrame, which is just a table of data, as you see.
So, now that we've got a DataFrame, we can see what we're working with, and when we ask…
when in jupyter we just put the name of a variable containing a DataFrame, we get the first five rows, the last five rows, and the size. So we've got 36,473 rows. Okay, so other
things I like to use for understanding a DataFrame is the “describe” method.
If you pass “include equals object” that will describe, that will describe, basically all the kind of the string fields, the non-numeric fields. So, in this case there's four of those,
and so you can see here that, that anchor field we looked at, there's actually only 733 unique values, okay, so this thing, you can see that there's lots of repetition out
of 36,000. So there's lots of repetition. This is the most common one: it appears 152
times. And then “context”, we also see lots of repetition – there's 106 of those contexts. So, this is a nice little method, we can see a lot about the data in a glance.
And when I first saw this in this competition I thought: well this is actually not that
much language data, when you think about it. The, you know… Each document is very short,
you know, three or four words really, and lots of it is repeated. So that's like…
as I'm looking through it I'm thinking, like, “what are some key features of this data set?”. And that would be something, I'd be thinking, well, that's, you know, we've
got to do a lot with not very much unique data here.
So here's how we can just go ahead and create a single string like I described which contains,
you know, some kind of field separator, plus the context, the target and the anchor. So
we're going to pop that into a field called “input”. Something slightly weird in pandas
is there's two ways of referring to a column. You can use square brackets and a string to get the input column or you can just treat it as an attribute. When you're setting it,
you should always use the form seen here (... df [‘input’]= …) When reading it you can use either. I tend to use this one because it's less typing.
So you can see now we've got this/these concatenated rows. So, head() is the first few rows.
So we've now got some documents to do NLP with. Now, the problem is, as you know from
the last lesson, neural networks work with numbers. All right, we're going to take some
Tokenization
numbers and we're going to multiply them by matrices, we're going to replace the negatives
with zeros and add them up, and we're going to do that a few times. That's our neural network. With some little wrinkles, but that's the basic idea. So how on earth do we do that
for these strings? So there's basically two steps we're going to take.
The first step is to split each of these into tokens. Tokens are basically words. We're
going to split it into words. There's a few problems with splitting things into words,
though. The first is that some languages like chinese don't have words, right, or at least
certainly not space separated words. And in fact in chinese it's sometimes… it's a bit fuzzy to even say where a word begins and ends. And some words are kind of not even…
the pieces are not next to each other. Another reason is that, what we're going to be doing is, after we've split it into words, or something like words, we're going to be getting a list
of all of the unique words that appear, which is called the vocabulary, and every one of those unique words is going to get a number. As you'll see later on the bigger the vocabulary,
the more memory is going to get used, the more data we'll need to train. In general
we don't want a vocabulary to be too big.
So instead, nowadays, people tend to tokenize into something called subwords which is pieces
of words – so I'll show you what it looks like. So the process of turning it into smaller
units like words, it's called tokenization – and we call them tokens instead of words. The token is just like the more general concept of, like, whatever we're putting it into.
So we're going to get Hugging Face transformers and Hugging Face datasets doing our work for
us, and so, what we're going to do is we're going to turn our pandas DataFrame into a
Hugging Face “datasets” Dataset. It's a bit confusing: pytorch has a class called
Dataset and Hugging Face has a class called Dataset and they're different things, okay, so this is a Hugging Face Dataset. “Hugging Face datasets” dataset. So we can turn a
DataFrame into a Dataset just using the from_pandas method and so we've now got a Dataset. So,
if we take a look it just tells us: all right it's got these features, okay? And remember
“input” is the one we just created with the concatenated strings and here's those 36,000 rows.
Okay, so now we're going to do these two things. Tokenization, which is to split each text
up into tokens, and the numericalization, which is to turn each token into its unique
id based on where it is in the vocabulary. The vocabulary, remember, being the unique,
the list of unique tokens. Now, particularly in this stage: tokenization, there's a lot
of little decisions that have to be made. The good news is you don't have to make them
because whatever pre-trained model you used the people that pre-trained it made some decisions,
and you're going to have to do exactly the same thing, otherwise you'll end up with a different vocabulary to them and that's going to mess everything up. So that means before
you start tokenizing you have to decide on what model to use. Hugging Face transformers
is a lot like “timm”. It has a library of, I believe, hundreds of models.
I guess I shouldn't say Hugging Face transformers. It's really the Hugging Face model hub. 44,000
Huggingface model hub
models, so even many more even than timm's image models. And so, these models, they vary
in a couple of ways. There's a variety of different architectures, just like in “timm”
but then something which is different to “timm” is that each of those architectures can be trained on different corpuses for solving different problems. So for example I could
type “patent” and see if there's any pre-trained patent: there is. Okay, so there's a patent,
there's a whole lot of pre-trained patent models. Isn't that amazing? So, quite often,
thanks to the Hugging Face model hub, you can start your pre-trained model with something
that's actually pretty similar to what you actually want to do, or at least was trained on the same kind of documents.
Having said that, there are some just generally pretty good models that work for a lot of
things a lot of the time, and deberta-v3 is certainly one of those. This is a very new
area. NLP has been, like, practically, really effective for, you know, general users, for
only a year or two, whereas for computer vision it's been quite a while. So you'll see, you'll
find that a lot of things aren't quite as well bedded down. I don't have a picture to
show you of which models are the best or the fastest and the most accurate and whatever,
right? This, a lot of this stuff is, like stuff that we're figuring out as a community
using competitions like this, in fact. And this is one of the first NLP competitions,
actually, in the kind of modern NLP era. So, you know, we've been studying these competitions
closely and yes, I can tell you that deberta-v3 is actually a really good starting point for
a lot of things so that's why we've picked it. So we pick our model and just like in “timm” for image, you know, models there's often going to be a small, a medium, a large
and of course we should start with small, right, because small is going to be faster to train we're going to be able to do more iterations….
and so forth. Okay.
So at this point remember the only reason we picked our model is because we have to make sure we tokenize in the same way.
To tell transformers that we want to tokenize the same way that the people that built a model did, we use something called AutoTokenizer. It's nothing fancy, it's basically just a
dictionary which says: “oh, which model uses which tokenizer?”. So when we say “AutoTokenizer.from_pretrained”
it will download the vocabulary and the details about how this particular model tokenized
the dataset. So, at this point we can now take that tokenizer and pass a string to it.
Examples of tokenized sentences
So, if I pass the string “G’day folks, I’m Jeremy from fast.ai!” you'll see it's
kind of putting it into words, kind of not. So if you've ever wondered whether “g'day”
is one word or two you know it's actually three tokens according to this tokenizer.
And “I'm” is three tokens. And “fast.ai” has three tokens. This punctuation is a token.
And so, you kind of get the idea. These underscores here? That represents the start of a word,
right. So that's kind of there's this concept that, like, the start of a word is kind of part of the token. So if you see a capital “I” in the middle of a word versus the
start of a word, that kind of means a different thing. So this is what happens when we tokenize
this sentence using the tokenizer that the deberta-v3 developers used.
So here's a less common (unless you're a big platypus fan like me), less common sentence.:
“A platypus is an ornithorhynchus anatinus”. So okay, in this particular vocabulary platypus
got its own word, its own token, but ornithorhynchus didn't. And so I still remember grade one,
for some reason our teacher got us all to learn how to spell “ornithorhynchus”, so, one of my favorite words. So you can see here it's been split into “_or”, “ni”,
“tho”, “rhynch”, “us”. So every one of these tokens you see here is going to be in the vocabulary, right? The
list of unique tokens that was created when this, when this particular model, this pre-trained
model, was first trained. So somewhere in that list we'll find “_A” (“underscore
capital A”), and it'll have a number and so that's how we'll be able to turn these
into numbers. So this first process is called tokenization and then the thing where we take
these tokens and turn them into numbers is called numericalization. So, our data set, remember we put our string into the “input” field so here's a function
Numericalization
that takes a document, grabs its input, and tokenizes it. Okay so we'll call this our
tokenization function. Tokenization can take a minute or two so we may as well get all
of our processes used doing it at the same time to save some time. So if you use the
“dataset dot map” it will parallelize that process, and just pass in your function. Make sure you pass “batched=True” so it can do a bunch at a time. Behind the scenes
this is going through something called the tokenizers library which is a pretty optimized Rust library that uses, you know, SIMD and parallel processing and so forth, so with
“batched=True” it'll be able to do more stuff at once. So look it only took six seconds,
so pretty fast. So now, when we look at a row of our tokenized data set, ~(it's going
to contain exactly the same as our original data set.) No sorry, it's not going to take
exactly the same as the original data set, it's going to contain exactly the same input as our original data set and it's also going to contain a bunch of numbers. These numbers
are the position in the vocabulary of each of the tokens in the string, so we've now
successfully turned a string into a list of numbers. That is a great first step. We can
see how this works, we can see for example that we've got “of” at this a separate
word, so that's going to be an “_of” in the vocabulary we can grab the vocabulary,
look up “_of”, find that it's 265 and check here: yep here it is 265. Okay, so it's
not rocket science right? It's just looking stuff up in a dictionary to get the numbers.
Okay, so that is the tokenization and numericalization necessary in NLP to turn our documents into
numbers to allow us to put it into our model. Any questions so far John?
Yeah, thanks Jeremy so there's a couple and this seems like a good time to throw them out – and it's related to how you've formatted your input data into these sentences that
Question: rationale behind how input data was formatted
you've just tokenized. So one question was really about: How you choose those keywords
and the order of the fields that you know, so I guess just interested in an explanation,
is it more art or science? how are you… No, it's arbitrary! I tried a few things I tried X, you know, I tried putting them backwards,
you know, doesn't matter! We just want some way, something that it can learn from, right?
So if I just concatenated it without these headers before each one, it wouldn't know
where “abatement of pollution” ended and where “abatement” started, right? So I did just something that I can learn from. This is a nice thing about neural nets, they're
so flexible… As long as you give it the information somehow, it doesn't really matter how you give it the
information, as long as it's there, right? I could have used punctuation, I could have put, like, I don't know, one semicolon here, and two here, and three here. Yeah it's not
a big deal. At the level where you're, like, trying to get an extra half a percent to get
up the leaderboard of a Kaggle competition you may find tweaking these things makes tiny differences, but in practice you won't generally find it matters too much.
Right, thank you. And I guess the second part of that, somebody's asking: If one of their
fields was particularly long, say it was a thousand characters, is there any special
handling required there? Do you need to re-inject those kinds of special marker tokens? Does
it change if you've got much bigger fields that you're trying to learn and query?
Yes. Long documents and ULMFiT require no special consideration. IMDb in fact has multi
thousand word movie reviews, and it works great. To this day, ULMFiT is probably the
ULMFit fits large documents easily
best approach for reasonably quickly and easily using large documents. Otherwise, if you use
transformer-based approaches, large documents are challenging. Specifically, transformers
basically have to do the whole document at once, whereas ULMFiT can split it into multiple pieces and read it gradually. So that means you'll find that people trying to work with
large documents tend to spend a lot of money on GPUs because they need the big fancy ones with lots of memory. So yes, generally speaking, I would say if you're trying to do stuff with
documents of over 2,000 words you might want to look at ULMFiT. Try transformers, see if
it works for you, but you know I'd certainly try both. For under 2,000 words, you know,
transformers should be fine unless you've got nothing but a laptop GPU, or something with not much memory.
So, Hugging Face transformers has these, you know… As I say it right now, I find them
somewhat obscure and not particularly well documented expectations about your data, that you kind of have to figure out, and one of those is that it expects that your target
is a column called “labels”. So once I figured that out, I just went, got our tokenized
DataSet, and renamed our “score” column to “labels”, and everything started working. I don't know if at some point they'll make this a bit more flexible, but it’s probably
best to just call your target “labels” and life will be easy. You might have seen back when I went “ls {path}” that there was another data set
there, called test.csv. And if you look at it, it looks a lot like our training set,
that's our other CSV that we've been working with, but it's missing the score. The labels.
This is called a test set – and so we're going to talk a little bit about that now
because my claim here is that perhaps the most important idea in machine learning is
the idea of having separate training, validation and test data sets.
Overfitting & underfitting
Test and validation sets are all about identifying and controlling for something called “overfitting”
and we're going to try and learn about this through example. This is the same information
that's in that Kaggle notebook – I've just put it on some slides here.
So I'm going to create a function here called plot_poly and I'm actually going to use the
same data that, I don't know if you remember, we used it earlier for trying to fit this
quadratic. We created some “x” and some “y” data. This is the data we're going
to use and we're going to use this to look at overfitting.
The details of this function don't matter too much, what matters is what we do with it, which is that it allows us to basically pass in the degree of a polynomial. So for
those of you that remember, a first degree polynomial is just a line, it's ”y = a x”.
A second degree polynomial will be ”y = ax^2 + bx + c”, third degree polynomial we'll
have a cubic, fourth degree you know quartic, and so forth. And what I've done here is I've
plotted what happens if we try to fit a line to our data. It doesn't fit very well. So
what happened here is we… we did a linear regression… and what we're using here is
a very cool library called scikit-learn. scikit-learn is something that, you know, I think it'd be fair to say it's mainly designed for kind of classic machine learning methods like,
kind of linear regression and stuff like that – I mean, very advanced versions of these things, but it's also great for doing these quick and dirty things. So in this case I
wanted to do a… what's called a polynomial regression… which is fitting the polynomial to data and it's just these two lines of code. It's a super nice library. So in this case,
a degree one polynomial is just a line, so I fit it, and then I show it with the data,
and there it is. Now that's what we call underfit, which is to say there's not enough, kind of,
complexity in this model I fit, to match the data that's there.
So an underfit model is a problem. It's got to be systematically biased, you know; all the stuff up here, we're going to be predicting too low; all the stuff down here, we're predicting
too low; all the stuff in the middle, we’ll be predicting too high. A common misunderstanding
is that simpler models are kind of more reliable in some way, but models that are too simple
will be systematically incorrect as you see here.
What happens if we fit a 10 degree polynomial?
That's not great either! In this case it's not really showing us what the actual… Remember
this was originally a quadratic. This is meant to match, right? And particularly at the ends here, it's predicting things that are way above what we would expect in real life right?
And it's trying to get… really it's trying really hard to get through this point, but clearly this point was just some noise, right? So this is what we call “overfit”. It's
done a good job of fitting to our exact data points, but if we sample some more data points
from this distribution, honestly we probably would suspect they're not going to be very close to this, particularly if they're a bit beyond the edges. So that's what overfitting
looks like. We don't want underfitting or overfitting. Now underfitting is actually pretty easy to recognize, because we can actually look at
our training data and see that it's not very close. Overfitting is a bit harder to recognize
because the training data is actually very close.
Now on the other hand, here's what happens if we fit a quadratic. And here I've got both
the real-line and the fit-line and you can see they're pretty close, and that's of course
what we actually want.
So how do we tell whether we have something more like this, or something more like this.
Well what we do is we do something pretty straightforward… is we take our original data set, these points, and we remove a few of them, so let's say 20% of them.
Splitting the dataset
We then fit our model using only those points we haven't removed, and then we measure how
good it is by looking at only the points we removed. So in this case let's say we had
removed… (I'm just trying to think) If I had removed this point here right, then it might have
kind of gone off down over here, and so then when we look at how well it fits we would say: oh! This one's miles away.
The model… the data that we take away and don't let the model see it when it's training
is called the “validation set.” So in fast.ai we've seen splitters before, right…
The splitters are the things that separate out the validation set. Fast.ai won't let you train a model without a validation set. Fast.ai always shows you your metrics, so
things like accuracy, measured only on the validation set. This is really unusual. Most
libraries make it really easy to shoot yourself in the foot, by not having a validation set,
or accidentally not using it correctly. So fast.ai won't even let you do that. So you've got to be particularly careful when using other libraries. HuggingFace transformers
is good about this, so they make sure that they do show you your metrics on a validation
set. Now creating a good validation set is not generally as simple as just randomly pulling
Creating a good validation set
some of your data out of your model, out of the data that you passed… that you train for your model. The reason why is… imagine that this was
the data you were trying to fit something to (okay) and you randomly remove some, so
it looks like this. That looks very easy doesn't it, because you've kind of like, still got
all the data you'd want around the points, and in a time series like this… this is dates and sales… in real life you're probably going to want to predict future dates. So
if you created your validation set by randomly removing stuff from the middle, it's not really a good indication of how you're going to be using this model. Instead you should truncate
and remove the last couple of weeks. So if this was your validation set and this is your
training set, that's going to be actually testing whether you can use this to predict
the future, rather than using it to predict the past.
Kaggle competitions are a fantastic way to test your ability to create a good validation
set, because Kaggle competitions only allow you to submit, generally, a couple of times
a day. The dataset that you are scored on in the leaderboard during that time is actually
only a small subset… in fact it's a totally separate subset… to the one you'll be scored on, on the end of the competition. And so most beginners on Kaggle overfit. And it's
not until you've done it that you'll get that visceral feeling of like: “oh my god, I
overfit.” In the real world outside of Kaggle you will often not even know that you overfit
– you just destroy value for your organization silently. So it's a really good idea to do
this kind of stuff on Kaggle a few times first, in real competitions, to really make sure that you are confident you know how to avoid overfitting – how to find a good validation
set and how to interpret it correctly. And you really don't get that until you screw it up a few times.
A good example of this was… there was a distracted driver competition on Kaggle – there
are these kind of pictures from inside a car, and the idea was that you had to try and predict
whether somebody was driving in a distracted way or not, and on Kaggle they did something
pretty smart… the test set, so the thing that they scored you on the leaderboard, contained people that didn't exist, at all, in the competition data that you train the model with. So if
you wanted to create an effective validation set in this competition, you would have to make sure that you separated the photos, so that your validation set contained photos
of people that aren't in the data you're training your model on.
There's another one like that, the Kaggle fisheries competition, which had boats that
didn't appear… so they were basically pictures of boats and you meant to try to guess/predict what fish were in the pictures. And it turned out that a lot of people accidentally figured
out what the fish were by looking at the boat, because certain boats tended to catch certain
kinds of fish. And so by messing up their validation set, they were really overconfident
of the accuracy of their model. I'll mention in passing, if you've been around Kaggle a bit, you'll see people talk about
cross-validation a lot. I'm just going to mention, be very very careful. Cross-validation
is explicitly not about building a good validation set, so you've got to be super super careful
if you ever do that.
Another thing I'll mention, is that scikit-learn conveniently offers something called train_test_split,
as does Hugging Face datasets, as does fast.ai – we have something called random splitter.
It can be encouraging… it can almost feel like it's encouraging you to use a randomized
validation set because there are these methods that do it for you. But yeah, be very very
careful, because very very often that's not what you want, okay. So we've learned what
a validation set is, so that's the bit that you pull out of your data that you don't train with, but you do measure your accuracy with. So what's a test set? It's basically another
validation set, but you don't even use it for tracking your accuracy while you build
Test set
your model. Why not? Well imagine you tried two new models every day for three months
(that's how long a Kaggle competition goes for.) So you would have tried 180 models,
and then you look at the accuracy on the validation set for each one. Some of those models you
would have got a good accuracy on the validation set, potentially because of pure chance, just
a coincidence, and then you get all excited and you submit that to Kaggle and you think you're going to win the competition, and you mess it up! And that's because you actually
overfit using the validation set. So you actually want to know whether you've really found a
good model or not. So in fact on Kaggle they have two test sets. They've got the one that
gives you feedback on the leaderboard during the competition and a second test set which
you don't get to see until after the competition is finished. So in real life you've got to
be very careful about this, not to try so many models during your model building process
that you accidentally find one that's good by coincidence. And only if you have a test
set that you've held out, or you know that. Now that leads to the obvious question which is very challenging, is you spent three months working on a model, worked well on your validation
set, you did a good job of locking that test set away in a safe so you weren't allowed to use it, and at the end of the three months you finally checked it on the test set, and
it's terrible. What do you do? Honestly you have to go back to square one. You know there
really isn't any choice other than starting again. So this is tough, but it's better to
know, right. Better to know than to not know, so that's what a test set is for.
Metric vs loss
So you've got a validation set, what are you going to do with it? What you're going to do with a validation set, is you're going to measure some metrics. So a metric is something
like “accuracy”. It's a number that tells you: How good is your model? Now on Kaggle
this is very easy. What metric should we use? Well they tell us… go to overview, click
on evaluation, and find out… and it says: oh, we will evaluate on the Pearson Correlation
Coefficient. Therefore this is the metric you care about So, one obvious question is: Is this the same as the loss function? Is this the thing that
we will take the derivative of, and find the gradient, and use that to improve our parameters
during training? And the answer is: maybe, sometimes, but probably not. For example,
consider accuracy. Now, if we were using accuracy to calculate our derivative and get the gradient,
you could have a model that's actually slightly better, you know, it's slightly like it's doing a better job of recognizing dogs and cats, but not so much better that it's actually
caused any incorrectly classified cat to become a dog. So the accuracy doesn't change at all.
So the gradient is zero. You don't want stuff like that. You don't want bumpy functions
because they don't have nice gradients – often they don't have gradients at all, they're basically zero nearly everywhere. You want a function that's nice and smooth. Something
like, for instance, the average absolute error, mean absolute error, which we've used before.
So that's the difference between your metrics and your loss. Now be careful, right, because when you're training your model's spending all of its time trying to improve the loss
and most of the time that's not the same as a thing you actually care about, which is your metric. So you've got to keep those two different things in mind.
The other thing to keep in mind is that in real life… you can't go to a website and
be told what metric to use. In real life… the model that you choose, there isn't one
The problem with metrics
number that tells you whether it's good or bad and even if there was you wouldn't be able to find it out ahead of time. In real life the model you use is a part of a complex
process often involving humans, both as users and customers and as people, you know, involved
in… as part of the process. There's all kinds of things that are changing over time
and there's lots and lots of outcomes of decisions that are made. One metric is not enough to
capture all of that. Unfortunately, because it's so convenient to pick one metric and
use that to say: I've got a good model, that very often finds its way into industry, into
government… where people roll out these things that are good on the one metric that
happened to be easy to measure. And again and again we found people's lives turned upside
down because of how badly they get screwed up by models that have been incorrectly measured
using a single metric. So my partner Rachel Thomas has written this article which I recommend
you read about “The problem with metrics is a big problem for AI”
It's not just an AI thing! There's actually this thing called “Goodhart’s Law” that states “when a measure becomes a target, it ceases to be a good measure.” The thing
is… so when I was a management consultant, you know, 20 years ago, we were always kind
of part of these strategic things trying to like: find key performance indicators and
ways to kind of, you know, set commission rates for sales people and we were really doing a lot of this, like, stuff which is basically about picking metrics and, you know,
we see that happen… go wrong in industry all the time. AI is dramatically worse because
AI is so good at optimizing metrics, and so that's why you have to be extra, extra, extra
careful about metrics, when you are trying to use a model in real life. Anyway, as I said in Kaggle, we don't have to worry about any of that, we're just going
to use the “Pearson correlation coefficient” which is all very well as long as you know what the hell the “Pearson correlation coefficient” is…
If you don't, let's learn about it. So “Pearson correlation coefficient” is usually abbreviated
Pearson correlation
using letter “r” and it's the most widely used measure of how similar two variables
are. And so, if your predictions are very similar to the real values then the “Pearson
correlation coefficient” will be high, and that's what you want. “r” can be between
minus-one and one. Minus-one means you predicted exactly the wrong answer, which in a Kaggle
competition could be great because then you can just reverse all of your answers and you'll be perfect. Plus-one means you've got everything exactly correct.
Generally speaking, in courses or textbooks, when they teach you about the “Pearson Correlation Coefficient”, at that point… at this point, they will show you a mathematical function.
I'm not going to do that because that tells you nothing about the “Pearson Correlation Coefficient.” What we actually care about is not the mathematical function, but how
it behaves; and I find most people, even who work in data science, have not actually looked
at a bunch of data sets to understand how “r” behaves. So let's do that right now
so that you're not one of those people. The best way I find to understand how data behaves in real life, is to look at real-life
data so there's a data set… scikit-learn comes with a number of data sets, and one of them is called “California housing” and it's a data set where each row is a district…
and, it's kind of demographic, sorry it's information… some demographic information
about different districts, and about the value of houses in that district.
I’m not going to try to plot the whole thing, it's too big, and this is a very common question
I have from people is: “how do I plot data sets with far too many points?” The answer
is very simple: get less points. So I just randomly grab a thousand points. Whatever
you see with a thousand points, is going to be the same as what you see with a million points. There's no point… no reason, to plot huge amounts of data generally just grab
a random sample. Now, numpy has something called “corecoeff()” to get the correlation coefficient between
every variable and every other variable, and it returns a matrix. So I can look down here,
and so for example, here is the correlation coefficient between variable one, and variable one. Which of course is exactly perfectly 1.0. Right? because variable one is the same
as variable one. Here is the small inverse correlation between variable one and variable
two, and medium-sized positive correlation between variable one and variable 3 and so
forth. This is symmetric about the diagonal because the correlation between variable 1
and variable 8 is the same as the correlation between variable 8 and variable 1. So this
is a correlation coefficient matrix.
So that's great when we wanted to get a bunch of values all at once. For the Kaggle competition we don't want that. We just want a single correlation number. If we just pass in a pair
of variables, we still get a matrix which is kind of weird.. it's kind of… it's not
weird, but it's not what we want! So we should grab one of these. So when I want to grab a correlation coefficient, I'll just return the zeroth row, first column. So that's what
“core” is. That's going to be our single correlation coefficient. So let's look at the correlation between two things; for example…
median income, and median house value: 0.67. Okay? Is that high? medium? low? How big is
that? What does it look like? So the main thing we need to understand is: what these things look like.
So what I suggest we do is: we're going to take a 10 minute break… nine minute break. We'll come back at half past, and then we're going to look at some examples of correlation
coefficients Okay. Welcome back! So what I've done here is… I've created a little function called
show correlations, and I'm passing a DataFrame and a couple of columns as strings. I'm going
to grab each of those columns as series, do a scatter plot, and then show the correlation.
So, we already mentioned “median income” and “median house value” of 0.68, so here
it is… here's what .68 looks like. So you know I don't know if you had some intuition about what you expected, but as you can see it's still plenty of variation, even at that
reasonably high correlation. Also, you can see here that visualizing your data is very important if you're working with
this data set, because you can immediately see all these dots along here. That's clearly
truncation right? So this is like, when... it's not until you look at pictures like this, that you're gonna pick stuff like this up. Pictures are great! Oh! little trick: on the
scatter plot, I put alpha as 0.5, that creates some transparency. For these kind of scatter
plots, that really helps, because it… like… kind of creates darker areas in places where
there's lots of dots. So, yeah, alpha in scatter plots is nice. Okay, here's another pair. So this one's gone down from 0.68 to 0.43. Median income versus
the number of rooms per house. As you'd expect more rooms… it's more income, but this is
a very weird looking thing. Now, you'll find that a lot of these statistical measures like
correlation rely on the square of the difference, and when you have big outliers like this,
the square of the difference goes crazy, and so this is another place we'd want to look at the data first, and say oh that's… that's going to be a bit of an issue. There's probably
Correlation is sensitive to outliers
more correlation here, but there's a few examples of some houses with lots and lots of rooms
where people that aren't very rich live. Maybe these are some kind of shared… shared accommodation or something?
So, “r” is very sensitive to outliers. So let's get rid of the houses… the rooms
with 15 rooms… the houses with 15 rooms or more, and now you can see it's gone up
from 0.43 to 0.68, even though we probably only got rid of one two three four five six
seven… got rid of seven data points! So we've got to be very careful of outliers! And that means if you're trying to win a Kaggle competition where the metric is correlation,
and you just get a couple of rows really badly wrong, then it's going to be a disaster to your score, right. So you've got to make sure that you do a pretty good job of every row.
So there's what a correlation of 0.68 looks like. Okay, here's a correlation of 0.34, and this is kind of interesting, isn't it? Because
0.34 sounds like quite a good relationship, but you almost can't see it! So this is something
I strongly suggest is, if you're working with a new metric, is: Draw some pictures of a
few different levels of that metric to kind of try to get a feel for… like… what does it mean? You know, what does 0.6 look like? What does 0.3 look like? And so forth.
And here's an example of a correlation of minus 0.2. This very slight, negative slope.
Okay, so there's just more of a kind of a general tip, of something I like to do when playing with a new metric, and I recommend you do as well. I think we've now got a sense
of what the correlation feels like. Now you can go look up the equation on Wikipedia if you're into that kind of thing.
We need to report the correlation after each epoch because we want to know how our training
is going. Hugging Face expects you to return a dictionary because it's going to use the
keys of the dictionary to like… label… each metric. So here's something that gets
the correlation, and returns it as a dictionary with the label “pearson”.
Okay, so we've done metrics, we've done our training/ validation split.
Oh! we might have actually skipped over the bit where we actually did the split! Did I?
I did! So, to actually do the split, because in this Kaggle competition – I've got another notebook,
we'll look at later, where we actually split this properly — but here we're just going to do a random split. Just to keep things simple for now, of 25 percent, will be…
of the data will be a validation set. So, if we go tok_ds.train_test_split() it returns
a data set dict; which has a train, and a test. So that looks a lot like a datasets
object in fast.ai. Very similar idea!
So this will be the thing that we'll be able to train with, so it's going to train with this data set, and return the metrics on this data set. This is really a validation set
but Hugging Face datasets calls it “test”.
Training a model
Okay. We're now ready to train our model. In fast.ai, we use something called a “learner.”
The equivalent in Hugging Face transformers is called “trainer”. So we'll bring that in; something we'll learn about quite shortly is the idea of “mini batches” and “batch
sizes.” In short, each time we pass some data to our model for training, it's going
to return… it's going to send through a few rows at a time to the GPU, so that it
can calculate those in parallel. Those… a bunch of rows is called a “batch” or
a “mini batch'' and the number of rows is called the “batch size.” So here, we're going to set the batch size to 128. Generally speaking, the larger your batch size, the
more it can do in parallel (at once) and it'll be faster, but if you make it too big you'll
get an “out of memory” error on your GPU. So, you know, it's a bit of trial and error
to find a batch size that works. “Epochs” we've seen before. Then we've got the “learning
rate.” We'll talk in the next lesson – unless we get to this lesson – about a technique
to automatically find a… or semi-automatically find a “good” learning rate. We already know what a learning rate is from the last lesson. I've played around and found one that
seems to train quite quickly without falling apart, so I just tried a few. Generally, I
kind of, you know, if I… if I don't have a so… Hugging Face transformers doesn't
have something to help you find the learning rate. This the integration we're doing in fast.ai, will let you do that, but if you're using a framework that doesn't have that,
you can just start with a really low learning rate, and then kind of double it, and keep doubling it until it falls apart.
Hugging Face transformers uses this thing called “training arguments” which is a class where you just provide all of the kind of configuration… so you have to…
tell it what your learning rate is. This stuff here is the same as what we call basically
fit_one_cycle() in fast.ai. You always want this to be true, because it's going to be faster… pretty much…
and then the… this stuff here, you can probably use exactly the same every time. There's probably a lot of boilerplate compared to fast.ai as you see. This stuff you can probably use the
same every time. Okay, so… We now need to create our model. So, the equivalent of the vision learner function that we've
used to automatically create a reasonable vision model? In Hugging Face transformers,
they've got lots of different ones depending on what you're trying to do. So, we're trying to do classification as we've discussed, of sequences, so if we call AutoModelForSequenceClassification,
it will create a model that is appropriate for classifying sequences from a train… pre-trained model, and this is the name of the model that we did earlier the deberta-v3.
It has to know when it adds that random matrix to the end, how many outputs it needs to have. So we have one label which is the score. So that's going to create our model, and then
this is the equivalent of creating a learner. It contains a model, and the data… the training
data, and the test data. Again, there's a lot more boilerplate here than fast.ai, but you can kind of see the same basic steps here. We just have to do a little bit more manually,
but it's not… you know, it's nothing too crazy. So, it's going to tokenize it for us using that function, and then these are the metrics… that will print out each time.
That's that little function we created which returns a dictionary.
At the moment I find Hugging Face transformers very verbose. It spits out lots and lots and lots of text which you can ignore, and we can finally call train, which will spit out
much more text again, which you can ignore, and as you can see, as it trains, it's printing
out the loss, and here's our Pearson correlation coefficient. So, it's training and we've got
a 0.834 correlation, that's pretty cool! Right. I mean it took what …? Oh here we are five
minutes to run. Maybe that's five minutes per epoch on Kaggle which doesn't have particularly great GPUs (but good for free) and we've got something that is you know got a very high
level of correlation in assessing how similar the two columns are, and the only reason it
could do that is because it used a pre-trained model, right. There's no way you could just
have that tiny amount of information and figure out whether those two columns are very similar.
This pre-trained model already knows a lot about language. It already has a good sense of whether two phrases are similar or not, and we've just fine-tuned it. You can see,
given that after one epoch it was already at 0.8. You know we… this was a model that already did something pretty close to what we needed. It didn't really need that much
extra tuning for this particular task. We got any questions there John? “Yeah we do! It's actually a bit back on the topic
Question: when is it ok to remove outliers?
before when you were showing us the visual interpretation of the Pearson Coefficient, and you were talking about outliers. And we've got a question here from Kevin, asking: how
do you decide when it's okay to remove outliers? Like, you… pointed out some in that data
set, and clearly your model is going to train a lot better if you clean that up; but I think
Kevin's point here is, you know, those kinds of outliers will probably exist in the test
set as well, so I think he's just looking for some practical advice on on how you handle that in a more general sense.”
So, outliers should never just be removed, like, for modeling… So if we take the example
of the California housing data set, you know, if I was really working with that dataset in real life, I would be saying, “oh that's interesting! it seems like there's a separate
group of districts with a different kind of behavior”. Yeah my guess is that they're going to be kind of like dorms or something like that, you know, probably low-income housing
and so I would be saying like, “oh clearly, from looking at this dataset, these two different
groups can't be treated the same way, they have very different behaviors, and I would probably split them into two separate analyses.
You know the... the word outlier... it kind of exists in a statistical sense, right? There
can be things that are well outside our normal distribution and mess up our kind of metrics and things. It doesn't exist in a real sense. It doesn't exist in a sense of like... oh...
things that we should, like, ignore or throw away. You know, some of the most useful kind of insights I've had in my life in data projects
has been by digging into outliers...so-called outliers... and understanding: well, what
are they? And where did they come from? …and it's kind of... often in those edge cases
that you discover really important things about, like, where processes go wrong – or
about, you know, kinds of behaviors you didn't even know existed, or indeed about, you know, kind of labeling problems or process problems which you really want to fix them at the source
because otherwise when you go into production you're going to have more of those so-called outliers. So yeah. I'd say never delete outliers without investigating them and having a strategy
for ...like... understanding where they came from and ...like... what should you do about them. All right. So now that we've got a trained model, you'll see that it actually behaves,
Predictions
you know, really a lot like a fast.ai learner and you know hopefully the impression you'll get from going through this process is largely a sense of familiarity, of like, “oh yeah
this looks like stuff I've seen before”, you know, like a bit more wordy and some slight
changes but it really is very very similar to the way we've done it before. Because now that we've got a trained trainer, rather than learner, we can call predict and now we're
going to pass in our dataset from the Kaggle test file – and so that's going to give
us our predictions, which we can cast to float.
And here they are. So here are the predictions we made of similarity. Now again, not just
for your inputs but also for your outputs, always look at them. Always. Right? And interestingly,
I looked at quite a few Kaggle notebooks from other people, for this competition, and nearly
all of them had the problem we have right now, which is negative predictions and predictions
over one. So I'll be showing you how to fix this in a more proper way, maybe hopefully
in the next lesson but for now you know we could at least just round these off ...right?
…because we know that none of the scores are going to be bigger than one or smaller than zero, so our correlation coefficient will definitely improve if we at least round
this up to zero and round this down to one. As I said, there are better ways to do this
but that's certainly better than nothing. So, in Pytorch, you might remember from when
we looked at ReLU, there's a thing called clip and that will clip everything under zero to zero and everything over one to one and so now that looks much better. So here's our
predictions. So Kaggle expects submissions to generally be in a CSV file and Hugging
Face datasets... it kind of looks a lot like pandas, really. We can create our submission
file from... with our two columns called dot csv… and there we go. That's basically it.
So yeah you know... it's... it's... it's kind of nice to see how... you know... it's a sense
how far deep learning has come since we started this course a few years ago. That nowadays
you know... there are multiple libraries around to kind of do this… the same thing. We can, you know, use them in multiple application areas. They all look kind of pretty familiar.
They're reasonably beginner friendly. And NLP, because it's kind of like the most recent
area that's really become effective in the last year or two, is probably where the biggest
opportunities are for, you know, big wins both in research and commercialization. And
Opportunities for research and startups
so if you're looking to build a startup, for example, one of the key things that VCs look for, you know, that they'll ask is like “oh why now?”.. you know... “why... why would
you build this company now?” And of course you know with NLP, the answer is really simple. It's like... it can often be like... “well until last year this wasn't possible” you
know, or “it took 10 times more time” or “it took 10 times more money” or whatever.
So I think NLP is a huge opportunity area.
It's worth thinking about both use and misuse of modern NLP, and I want to show you a subreddit.
Misusing NLP
Here is a conversation on a subreddit from a couple of years ago. I'll let you have a quick read of it.
So the question I want you to be thinking about is: What subreddit do you think this comes from? …this debate about military spending?
And the answer is it comes from a subreddit that posts automatically generated conversations between GPT2 models. Now this is like a totally previous generation of model – they're much
much better now – so even then you could see these models were generating context appropriate
believable prose.
You know I would strongly believe that like any of our... kind of like... upper tier of
competent fast.ai alumni would be fairly easily able to create a bot which could create context
appropriate prose on twitter or facebook groups or whatever, you know, arguing for a side
of an argument and you could scale that up such that 99% of twitter was these bots and
nobody would know. You know, nobody would know. And that's very worrying to me because
a lot of, you know, a lot of...kind of... the way people see the world is now really
coming out of their... their social media conversations, which at this point they're
controllable. Like... it would not be that hard to create something that's kind of optimized
towards moving a point of view amongst a billion people, you know, in a very subtle way, very
gradually, over a long period of time by multiple bots each pretending to argue with each other and one of them getting the upper hand and so forth.
Here is the start of an article in the Guardian which I'll let you read.
This article was, you know, quite long. These are just the first few paragraphs and at the end, it explains that this article was written by GPT3. It was given the instruction “please
write a short op-ed around 500 words. Keep the language simple and concise. Focus on why humans have nothing to fear from AI.” So GPT3 produced eight outputs and then they
say, basically the... the editors at The Guardian did about the same level of editing that they would do for humans. In fact, they found it a bit less editing required than humans. So
you know again, like, you can create longer pieces of context appropriate prose designed
to argue a particular point of view. What kind of things might this be used for? You
know, that we won't know probably for decades if ever but sometimes we get a clue based
on older technology. Here's something from back 2017 and the pre ...kind of... deep learning
NLP days. There were millions of submissions to the FTC about the net neutrality situation
in America. Very very heavily biased towards the point of view of saying “we want to
get rid of net neutrality”. An analysis by Jeff Kao showed that something like 99%
of them and in particular, nearly all of the ones which were pro removal of net neutrality,
were clearly auto generated by basically ...if you look at the green, there's like, selecting
from a menu. So we've got... Americans as opposed to Washington bureaucrats deserve to enjoy the services they desire… individuals as opposed to Washington bureaucrats should
be blah...blah... people like me as opposed to so-called experts... and you get the idea. Now this is an example of a very very, you know, simple approach to auto-generating huge
amounts of text. We don't know for sure but it looks like this might have been successful
because this went through. You know, despite what seems to be actually overwhelming disagreement
from the public that everybody, almost everybody, likes net neutrality, the FTC got rid of it
and this was a big part of the basis. It’s like “oh we got all these comments from the public and everybody said they don't want net neutrality”. So imagine a similar thing
where you absolutely couldn't do this. You couldn't figure it out because everyone was really very compelling and very different. That's, you know, it's kind of worrying about
how we deal with that. You know, I will say... when I talk about this stuff, often people
say “ah no worries we'll build a model to recognize... you know... bot generated content”
but, you know, if I put my black hat on, I'm like “no that's not gonna work, right?”. If you told me to build something that beats the bot classifiers, I'd say “no worries,
easy. You know, I will take the... the code or the service... or service... whatever that does the bot classifying and I will include beating that in my loss function and I will
fine-tune my model until it beats the bot classifier. You know, when I used to run an
email company, we had a similar problem with spam prevention and our spammers could always
take a spam prevention algorithm and change their emails until it didn't get the spam
prevention algorithm anymore, for example. So yes, I... I'm really excited about the
opportunities for... for students in this course to build, you know, I think very valuable
businesses, really cool research and so forth using these pretty new NLP techniques that
are now pretty accessible and I'm also really worried about the things that might go wrong. I do think though that the more people that understand these capabilities the less chance
they'll go wrong. John, was there some questions? “Yeah I mean it's a throwback to the... to the workbook that you had before... yeah
Question: isn’t the target categorical in this case?
that's the one. The question Manikandan is asking... shouldn't num labels be five zero
zero point two five zero point five zero point seven five one instead of one? Isn't the target
a categorical, or are we considering this as a regression problem?” Yeah it's a good question. So there's one label because there's one column. Even if
this was being treated as a categorical problem with five categories, it's still considered one label.
In this case though, we're actually treating it as a regression problem. It's one of the
things that's a bit tricky. I was trying to figure this out just the other day. It's not documented as far as I can tell but on the Hugging Face transformers website... but if
you pass in one label to AutoModelForSequenceClassification, it turns it into a regression problem which
is actually why we ended up with predictions that were less than zero and bigger than one.
So we'll be learning next time about the use of sigmoid functions to resolve this problem
and that should fix it up for us Okay, great. Well thanks everybody. I hope you enjoyed learning about NLP as much as
I enjoyed putting this together. I'm really excited about it and can't wait for next week's
lesson. See ya!

---

# 5

OK, hi everybody, and welcome to Practical  Deep Learning for Coders, lesson five.  
We're at a stage now where we're going to be  getting deeper and deeper into the details of how  
these networks actually work. Last week we saw how  to use a slightly lower level library than fastai,  
being Hugging Face Transformers, to train a pretty  nice NLP model. And today we're going to be going  
back to tabular data and we're going to be trying  to build a tabular model, actually from scratch.
We're going to build a couple of different  types of tabular models from scratch.   So the problem that I'm going to be  working through is the Titanic problem,  
which if you remember back a couple of weeks, is  the data set that we looked at on Microsoft Excel.  
And it has each row as one passenger on the  Titanic, so this is a real world data set,  
a historic dataset. Tells you both that passenger  survived, what class they were on in the ship,  
their sex, age, how many siblings,  how many other family members,   how much they spent on fare and whereabouts  they embarked from the three different cities.  
You might remember that we built a linear  model, we then did the same thing using  
matrix multiplication and we also created  a very very simple neural network.  
You know Excel can do nearly  everything we need as you saw,   to build a neural network, but it starts to  get unwieldy and so that's why people don't use  
Excel for neural networks in practice  – instead we use a programming language   like Python. So what we're going to do today,  is we're going to do the same thing with Python,  
so we're going to start working through the  linear model and neural net from scratch   notebook which you can find on Kaggle or on the  course repository. And today what we're going  
Linear model and neural net from scratch
to do is we're going to work through the one in  the clean folder so both for fastbook (the book)  
and course 22 (these lessons) the clean folder  contains all of our notebooks but without  
any prose or any outputs. So here's  what it looks like when I open up   the linear model and neural net from  scratch, in Jupyter. What I'm using here  
is Paperspace Gradient, which as I mentioned  a couple of weeks ago, is what i'm going to be  
doing most things in – it looks a little bit  different to the normal Paperspace Gradient  
because the the default view  for Paperspace Gradient,  
at least as I do this course, is  their rather awkward notebook editor  
which at first glance has the same features  as the the real Jupyter Notebook and Jupyter  
Lab environments. But, in practice,  it’s actually missing lots of things.  
So this is the normal Paperspace, so  remember you have to click this button  
and the only reason you might keep this  window running – is then you might go over   here to the machine to remind yourself when  you close the other tab to click stop machine.  
If you're using the free one it doesn't  matter too much and also when I start it I   make sure I've got something to set to just  shut down automatically in case I forget.
So other than that we're going to…  we can stay in this tab – and because   this is Jupyter Lab that it (Paperspace  Gradient) runs and you can always  
switch over to classic Jupyter  Notebook if you want to.
So given that they've kind of got tabs inside  tabs, I normally maximize it at this point.  
And it's really good and really helpful to  know the keyboard shortcuts – so ctrl shift   square bracket right and left switch between tabs  – that's one of the key things to know about.  
Okay, so I've opened up the clean version of the  linear model and neural net from scratch notebook.  
And so remember when you go back through  the video, kind of the second time  
or through the notebook a second time, this is  generally what you want to be doing – is going   through the clean notebook and before you run each  cell try to think about like oh what did jeremy  
say ‘why are we doing this? what output would  I expect?’ make sure you get the output you'd  
expect. And if you're not sure why something  is the way it is, try changing it and see what  
happens. And then if you're still not sure ‘well  why did that thing not work the way I expect?’,   you know, search the forums if anybody's asked  that question before and you can ask the question  
on the forum yourself if you're still not sure.  So as I think we've mentioned briefly before,  
I find it really nice to be able to use the same  notebook both on Kaggle and off Kaggle – so most  
of my notebooks start with basically the  same cell, which is something that just   checks whether we're on Kaggle. So Kaggle sets an  environment variable, so we can just check for it,  
that way we know if we're on kaggle and so then  if we are on kaggle, you know a notebook that's  
part of a competition will already have  the data downloaded and unzipped for you,   otherwise if I haven't downloaded the data before,  then I need to download it and unzip it. Okay so  
Kaggle is a pip installable module so you type  “pip install kaggle”. If you're not sure how to  
do that – you should check our deep dive lessons  to see exactly the steps – but roughly speaking  
you can use your console, pip install and whatever  you want to install. Or, as we've seen before,  
you can do it directly in a notebook by  putting an exclamation mark at the start.  
So that's going to run not python, but  a shell command. Okay so that's enough  
to ensure that we have the data downloaded and  a variable called path that's pointing at it.  
Most of the time we're going to be using at least  Pytorch and Numpy, so we import those so they're  
available to Python. And when we're working  with tabular data, as we've talked about before,  
we're generally also going to want to use  pandas. And it's really important that you're   somewhat familiar with the kind of basic API of  these three libraries and I've recommended Wes  
Mckinney's book before, particularly for these  ones, (https://wesmckinney.com/pages/book.html). One thing just by the way is that these things  tend to assume you've got a very narrow screen,  
which is really annoying, because it always wraps  things. So if you want to put these three lines as   well – then it just makes sure that everything  is going to use up the screen properly. Okay,  
so as we've seen before, you can read a  comma separated values file with pandas  
Cleaning the data
and you can take a look at the first few  lines in the last few lines and how big it is.   And so here's the same thing as our spreadsheet.
Okay so there's our data from the  spreadsheet and here it is as a data frame. 
So, if we go df.isna():  
that returns a new data frame – in which every  column tells us whether or not that particular  
value is nan. So ‘nan’ is ‘not a number’ and the  most common reason you get that is because it was  
missing. Okay so a missing value is obviously  not a number. So in the Excel version we did  
something you should never usually do – we  deleted all the rows with missing data –  
just because in Excel it's a little bit harder to  work with. In pandas it's very easy to work with.   First of all, we can just sum up what I just  showed you. Now if you call “sum” on a dataframe,  
it sums up each column. So you can see that  there's, kind of, some small foundational concepts  
in pandas – which, when you put them together,  take you a long way. So one idea, is this idea,   that you can call a method on a dataframe and  it calls it on every row. And then you can call  
a reduction on that and it reduces each column.  So now we've got the total and in Python, Pandas,  
Numpy and Pytorch you can treat a boolean as a  number and true will be one false will be zero.  
So this is the number of missing values in each  column. So we can see that ‘cabin’, out of 891  
rows, it's nearly always empty. ‘Age’ is empty a  bit of the time. ‘Embarked’ is almost never empty.  
So if you remember from Excel we need to multiply  a coefficient by each column – that's how we  
create a linear model. So how would you multiply  a coefficient by a missing value? You can't.
There's lots of ways of (it's called imputing  missing values) replacing missing value with   a number. The easiest, which always works,  is to replace missing values with the mode  
of a column. The mode is the most common value.  That works for both the categorical variables,  
it's the most common category; And for  continuous variables, that's the most common  
number. So you can get the mode by calling  “df.mode”. One thing that's a bit awkward  
is that if there's a tie, for the mode –  so there's more than one thing that's the  
most common – it's going to return multiple  rows. So I need to return the zeroth row.  
So here is the mode of every column. So we  can replace the missing values for ‘age’  
with 24 and the missing values for ‘cabin’  with ‘b96’, ‘b98’ and ‘embarked’ with ‘s’.
I'll just mention in passing, I am not going to  describe every single method we call in every  
single function we use. And that is not because  you're an idiot if you don't already know them,  
nobody knows them. Alright, but I don't know  which particular subset of them you don't know.  
So let's assume just to pick a number at  random, that the average fastai student knows  
80% of the functions we call. Then I  could tell you what every function is,  
in which case 80% of the time I'm wasting  your time because you already know.  
Or I could pick 20% of them at random, in which  case I'm still not helping, because most of the   time it's not the ones you don't know. My approach  is that for the ones that are pretty common,  
I’m just not going to mention it at all, because  I'm assuming that you'll google it. So it's really   important to know, for example, if you don't know  what “iloc” is: that's not a problem, it doesn't  
mean you're stupid, right?, it just means you  haven't used it yet and you should google it.
So I’ll mention in this particular case, this  is one of the most important pandas methods,  
because it gives you the row located at this  index: i’th index and lock for location,  
so this is the zero’th row. But yeah, I do  kind of go through things a little bit quickly  
on the assumption that students, fast.ai  students are, you know, proactive-curious people.  
And if you're not a proactive-curious person  then you could either decide to become one   for the purpose of this course, or maybe this  course isn't for you. All right, so, a DataFrame  
has a very convenient method called fillna() and  that's going to replace the not-a-numbers with  
whatever I put here and the nice thing about  pandas is it kind of has this understanding  
that columns match to columns so, it's  going to take the mode from each column  
and match it to the same column in the  DataFrame and fill in those missing values.  
Normally that would return a new  DataFrame; many things including   this one in pandas have an in-place argument  that says: actually modify the original one,  
and so if I run that. Now if I call .isna.sum()  they’re all 0. So that's like the world's simplest  
way to get rid of missing values. Okay so,  why did we do it the world's simplest way?
Because, honestly, this doesn't make much  difference most of the time and so I'm not going  
to spend time the first time I go through and  build a baseline model doing complicated things  
when I don't necessarily know  that I need complicated things.   And so imputing missing values is an  example of something that, most of the time,  
this dumb way – which always works  without even thinking about it – will be   quite good enough you know for nearly all the  time. So we keep things simple where we can.
John!, question Jeremy, we've got a question on this topic,  
Javier is sort of commenting on the assumption  involved in substituting with the mode  
and he's asking: in your experience,  what are the pros and cons of doing this   versus, for example, discarding ‘Cabin’ or  ‘Age’ as fields that we even train the model? 
Yeah, so I would certainly never throw them out,  right? There's just no reason to throw away data  
and there's lots of reasons to not  throw away data. So, for example,   when we use the fast.ai library – which  we'll use later – one of the things  
it does – which is actually a really  good idea – is it creates a new column   for everything that's got missing values, which is  boolean, which is: did that column have a missing  
value for this row? And so maybe it turns out  that ‘Cabin’ – being empty – is a great predictor,  
so yeah, I don't throw out rows  and I don't throw out columns.
Okay So, it's helpful to understand a bit more  about our data set and a really helpful...  
I've already imported this. A really helpful,  you know, quick method and again, it's kind of  
nice to know like a few quick things you can do  to get a picture of what's happening in your data  
is describe(). And so describe; you can say: okay,  describe all the numeric variables, and that gives  
me a quick sense of what's going on here; so we  can see ‘Survived’ clearly is just zeros and ones  
–because all of the quartiles are zeros and  ones–. Looks like ‘Pclass’ is one, two, three…
What else do we see? ‘Fair’ is an interesting  one, right?, lots of smallish numbers and one   really big number –so probably long tailed–.  So it's, yeah, good to have a look at this to  
see what's going on for your numeric variables.  So as I said ‘Fair’ looks kind of interesting.  
To find out what's going on there I would  generally go with a histogram so, if you   can't quite remember what a histogram is, again,  google it but, in in short, it shows you: for  
each amount of ‘Fare’, how often does that ‘Fare’  appear and it shows me here that the vast majority  
of ‘Fare’ are less than 50 dollars but there's  a few right up here to 500. So this is what we  
call a long tail distribution, a small number  of really big values and lots of small ones.
There are some types of model which  do not like long tail distributions.  
Linear models is certainly one of them and neural  nets are generally better behaved without them as  
well. Luckily there's an almost surefire way  to turn a long tail distribution into a more  
reasonably centered distribution: and that  is to take the log. We use logs a lot in  
machine learning. For those of you that  haven't touched them since year 10 math,  
it would be a very good time to, like, go to Khan  Academy or something and remind yourself about   what logs are and what they look like, because  they're actually really, really important.  
But the basic shape of the log  curve causes it to make, you know,   really big numbers less really big and doesn't  change really small numbers very much at all,  
so if we take the log… now log of 0  is NaN so a useful trick is to just  
do log(x + 1) and in fact there is a log p1 if  you want to do that, it does the same thing.
So, if we look at the histogram of  that, you can see it's much more,   you know, sensible now. It's kind of centered  and it doesn't have this big long tail,  
so that's pretty good. So we'll be  using that column in the future.
As a rule of thumb, stuff  like money or population,  
things that kind of can grow exponentially, you  very often want to take the log of. So if you have  
a column with a dollar sign on it, that's a good  sign it might be something to take the log of.
So there was another one here,  which is, we had a numeric   which actually doesn't look numeric at all,  it looks like it's actually categories.  
So pandas gives us a .unique(). And so we can  see, yep, they're just one, two and three,  
are all the levels of ‘Pclass’: that's their  first class, second class, or third class.  
We can also describe all  the non-numeric variables.   And so we can see here that, not  surprisingly, names are unique  
because the count of names is the same  as count unique. There's two sexes,  
681 different tickets, 147 different  cabins and three levels of ‘Embarked’.
So, we cannot multiply the  letter “S” by a coefficient. 
Or the word “male” by a coefficient. 
So, what do we do?   What we do is we create something called  dummy variables. Dummy variables are…  
and we can just go get_dummies() a column that  says, for example, is ‘Sex’ female, is ‘Sex’ male,  
is ‘Pclass’ 1, is ‘Pclass’ 2, is ‘Pclass’ 3…  So, for every possible level, of every possible   categorical variable, it's a boolean column of  did that row have that value of that column.  
So I think we've briefly talked about this  before, that there's a couple of different   ways we can do this: one is that for n… an n level  categorical variable, we could use n-1 levels.  
In which case we also need a constant term in our  model. Pandas, by default, shows all n levels,  
although you can pass an argument to change  that if you want. Oh yeah, ‘drop_first’.
I kind of like having all of them  sometimes because then you don't   have to put in a constant term and it's a bit  less annoying and it can be a bit easier to  
interpret but I don't feel  strongly about it either way. 
Okay, so here's a list of all of the columns  that pandas added. I guess directly speaking  
I probably should have automated that, but never  mind, I just copied and posted them. And so here  
are a few examples of the added columns. In Unix,  pandas, lots of things like that, ‘head’ means  
the first few rows, or the first few lines. So,  five by default in pandas, so here you can see  
they're never both male and female, they're  never neither, they're always one or the other.
All right, so, with that now we've got  numbers, which we can multiply by coefficients.  
It's not going to work for   ‘Name’, obviously, because we'd have 891 columns  and all of them would be unique. So we'll ignore  
that for now. That doesn't mean it's… have to  always ignore it and in fact something I did do…  
something I did do on the forum topic, because I  made a list of some nice Titanic notebooks that  
I found and quite a few of them really go hard  on this name column and in fact one of them,  
yeah, this one… In what I believe is, yes, Chris Deotte first  ever Kaggle notebook. He's now the number one  
ranked Kaggle notebook person in the  world, so this is a very good start.  
He got a much better score than any model  that we're going to create in this course,   using only that column: ‘Name’. And basically,  yeah, he came up with this simple little  
decision tree, by recognizing, you know, all  of the information that's in a ‘Name’ column.
So, yeah, we don't have to   treat, you know, a big string of letters  like this as a random big string of letters,  
we can use our domain expertise to recognize that  things like “Mr” have meaning and that people with  
the same surname might be in the same family  and actually figure out quite a lot from that.  
But that's not something I'm going to do, I'll let  you look at those notebooks if you're interested   in the feature engineering and I do  think that they're very interesting so  
do check them out. Our focus today is on building  a linear model and a neural net from scratch,  
not on tabular feature engineering, even  though that's also a very important subject.
Okay so, we talked about how matrix  modification makes linear models much easier  
and, the other thing we did in Excel  was element-wise multiplication.   Both of those things are much easier if  we use Pytorch instead of plain Python,  
or we could use Numpy, but I tend to just  stick with Pytorch when I can because it's   easier to learn one library than two.  So I just do everything in Pytorch,  
I almost never touch Numpy nowadays –they're  both great– but they do everything each other  
does except Pytorch also does differentiation  and GPUs so why not just learn Pytorch.
So, to turn a column into something that I can do  
Pytorch calculations on, I have to turn it into  a tensor. So a tensor is just… it's what num…  
numpy calls an array. It's what mathematicians  will call either a vector or a matrix or once  
we go to higher ranks, mathematicians  and physicists just call them tensors.
In fact this idea originally in computer science  came from a notation developed in the 50s called  
APL, which was turned into a programming language  in the 60s by a guy called Ken Iverson. And Ken  
Iverson actually came up with this idea from, he  said, his time doing tensor analysis in Physics.  
So there… these areas are very related. So we  can turn the “Survived” column into a tensor,  
and we'll call that tensor our dependent variable.  That's the thing we're trying to predict. Okay,  
so now we need some independent variables. So  our independent variables are ‘Age’, ‘Siblings’…
That one is… Oh yeah, the number of the family members.
Uh… the log of ‘Fare’ that we just created,  plus all of those dummy columns we added,  
and so we can now grab those  values and turn them into a tensor.  
And we have to make sure they're floats. We  want them all to be the same data type, and  
Pytorch wants things to be floats, if you're going  to multiply things together. So there we are.
And so one of the most important attributes of  a tensor, probably the most important attribute,  
is its shape, which is how many rows does  it have, and how many columns does it have.
The length of the shape…
is called its rank. That's the rank of the tensor.  It's the number of dimensions or axes that it has.  
So a vector is length of rank one, a matrix is  rank two, a scalar is rank zero, and so forth.  
I try not to use too much jargon, but there's  some pieces of jargon that are really important,  
because you, like otherwise, you're gonna have  to say the length of the shape again and again.   It's much easier to say rank, so we'll say… we'll  use that word a lot. So a table is a rank two  
tensor. Okay, so we've now got the data in  good shape. Here's our independent variables,  
Setting up a linear model
and we've got our dependent variable. So we can  now go ahead and do exactly what we did in Excel…
which is to multiply our rows of data…
By these coefficients. And remember to start with,  we create random coefficients. So we're going  
to need one coefficient for each column. Now in  Excel we also had a constant, but in our case now,  
we've… we've got every column, every level in  our dummy variable, so we don't need a constant.  
So the number of coefficients we need is equal  to the shape of the independent variables,  
and it's the index one element. That's the  number of columns. That's how many coefficients  
we want. So we can now…now… ask Torch… Pytorch  to give us some random numbers, n_coeff of them.  
They're between zero and one, so if we  subtract a half(0.5), then they'll be centered.  
And there we go. Before I do that,  I set the seed. What that means is  
in computers… computers in general  cannot create truly random numbers.
Instead they can calculate a sequence of numbers  that behave in a random-like way. That's actually  
good for us, because often in my teaching I  like to be able to say, you know, in the prose,   “oh look, that was two, now it's three”,  or whatever. And if I was using really  
random numbers, then I couldn't do that  because it'd be different each time. So   this is… makes my results reproducible.  That means if you run it, you'll get the  
same random numbers as I do, by saying “start  the pseudo random sequence with this number”.
I’ll mention in passing, a lot of people are very,  very into reproducible results. They think it's  
really important to always do this. I strongly  disagree with that. In my opinion, an important  
part of understanding your data is understanding  how much it varies from run to run. So if I'm not  
teaching, and wanting to be able to write things  about these pseudo-random numbers, I almost never  
use a manual seed. Instead, I like to run things  a few times, and get an intuitive sense of like…  
“oh, this is like… very very stable” or “oh, this  is all over the place”. I'm getting an intuitive  
understanding of how your data behaves, and  your model behaves, is really important.  
Now here's one of the coolest lines of code  you'll ever see. I know it doesn't look like much,  
but think about what it's doing, yeah.
Okay so, we've multiplied a matrix by a  vector. Now that's pretty interesting.  
Now mathematicians amongst you will know that  you can certainly do a matrix-vector product,  
but that's not what we've done here at all. We've  used element-wise multiplication. So normally,  
if we did the element-wise multiplication  of two vectors, it would multiply,  
you know… element one with element one, element  two with element two, and so forth. And create  
a vector of the same size output. But here,  we've done a matrix times a vector. How does  
that work? This is using the incredibly powerful  technique of broadcasting, and broadcasting again  
comes from APL, a notation invented in the 50s  in a programming language developed in the 60s.
And it's got a number of benefits. Basically,  what it's going to do is it's going to take each  
coefficient and multiply them in turn by every  row in our matrix. So if you look at the shape…
of our independent variable, and the shape of  our coefficients, you can see that each one of  
these coefficients can be multiplied  by each of these 891 values in turn.
And so the reason we call it broadcasting  is it's as if this is 891 columns by 12…  
891 rows by 12 columns. It's as  if this was broadcast 891 times.  
It's as if we had a loop, looping 891 times,  and doing, coefficients times row zero,  
coefficients times row one, coefficients times row  two, and so forth. Which is exactly what we want.  
Now, reasons to use broadcasting… obviously,  the code is much more concise. It looks more  
like math, rather than clunky programming  with lots of boilerplate. So that's good.  
Also, that broadcasting all happened in optimized  C code, and if, in fact, it's being done on a GPU,  
it's being done in optimized GPU assembly CUDA  code. It's going to run very, very fast indeed.  
And this is a trick of why we can use  a so-called slow language like Python   to do very fast big models. It’s because a  single line of code like this can run very  
quickly on optimized hardware, on lots and  lots of data. The rules of broadcasting are  
a little bit subtle, and important to know.  And so I would strongly encourage you…
to google “numpy broadcasting rules”.
And see exactly how they work. But you know,  the kind of intuitive understanding of them,   hopefully, you'll get pretty quickly, which  is, generally speaking, you can kind of,  
as long as the last axes match, it'll broadcast  over those axes. You can broadcast a rank-three  
thing with a rank-one thing, or you know, most  simple version would be tensor one, two, three,
times two. So broadcast a…
scalar over a vector. That's exactly what  you'd expect. So it's copying effectively  
that 2.0 into each of these parts,  multiplying them together. But it   does… it doesn't use up any memory to do that.  It's kind of a virtual copying, if you like.  
So this line of code “independent” by  “coefficients”, is very, very important.   And it's the key step that we wanted to  take, which is, now we know exactly how…  
what happens when we multiply the  coefficients in. And if you remember back to Excel, we did that product and then in Excel  there's a sum product we then added it all  
together, because that's what a linear model  is: it's the coefficients times the values  
added together. So we're now going  to need to add those together.  
But before we do that, if we did add up this  row, you can see that the very first value  
has a very large magnitude and all the other ones  are small same with row two, same with row three,  
same with row four. What's going on here?, well  what's going on is that the very first column  
was ‘Age’. And ‘Age’ is much bigger  than any of the other columns.  
It's not the end of the world but it's not ideal,  right?, because it means that a coefficient of,  
say, 0.5 times ‘Age’ means something  very different to a coefficient of say  
0.5 times ‘LogFair’, right? And that means  that that random coefficient we start with,  
it's going to mean very different things for  different columns and that's going to make   it really hard to optimize. So we would like  all the columns to have about the same range.
So, what we could do…  
as we did in Excel, is to divide them  by the maximum, so the maximum… so we   did it for ‘Age’ and we also did it for  ‘Fair’ –in this case i didn't use log.
So we can get the max of each row by calling  .max() and you can pass in a dimension –do you  
want the maximum of the rows or the maximum of  the columns?– we want the maximum over the rows,  
so we pass in dimension zero (dim=0). So those  different parts of the shape are called either  
axes or dimensions, Pytorch calls some dimensions.  So that's going to give us the maximum of each row  
and if you look at the docs for Pytorch's max  function it'll tell you it returns two things:  
the actual value of each maximum and the index  of where which row it was. We want the values.  
So now thanks to broadcasting we can  just say take the independent variables   and divide them by the vector of  values, again, we've got a matrix  
and a vector and so this is going to do an  element-wise division of each row of this,  
divided by this vector. Again,  in a very optimized way.  
So if we now look at our normalized  independent variables by the coefficients,  
you can see they're all pretty  similar values, so that's good. There's lots of different ways of normalizing  but the main ones you'll come across is either  
dividing by the maximum or subtracting the  mean and dividing by the standard deviation,  
it normally doesn't matter too much.  Because I'm lazy I just pick the easier one  
and being lazy and picking the easier  one is a very good plan in my opinion.   So now that we can see that multiplying  them together is working pretty well,  
we can now add them up and now we  want to add up over the columns…
and that would give us predictions. Now obviously  just like in Excel, when we started out, they're   not useful predictions, because they're random  coefficients, but they are predictions nonetheless  
and here's the first 10 of then. So then remember, we want to use gradient  descent to try to make these better.  
So to do gradient descent we need a loss, right?,  the loss is the measure of how good or bad  
are these coefficients. My favorite  loss function, as a kind of like:  
don't think about it, just chuck something  out there, is the mean absolute value   and here it is torch dot absolute value of  the error, the difference, take the mean.
And often, stuff like this, you'll see people will  use pre-written, mean absolute error functions,  
which is also fine but I quite like to write it  out because I can see exactly what's going on,  
no confusion, no chance of misunderstanding.  So those are all the steps I'm going to need to  
create coefficients, run a linear model, and get  its loss. So, what I like to do in my notebooks,  
like not just for teaching but all the time, is to  like do everything step by step manually and then  
just copy and paste the steps into a function.  So here's my calc_preds() function, is exactly  
Creating functions
what I just did, right? Here's my calc_loss()  function, exactly what I just did. And that way,  
you know, I… a lot of people like go back and  delete all their explorations, or they like  
do them in a different notebook, or they're like  working in an IDE, they'll go and do it in some,   you know, line oriented REPL or whatever but if,  you know, think about the benefits of keeping it  
here when you come back to it in six months,  you'll see exactly why you did what you did   and how we got there, or if you're showing it to  your boss or your colleague you can see, you know,  
exactly what's happening, what does each step look  like, I think this is really very helpful indeed.  
I know not many people code that  way, but I feel strongly that it's   a huge productivity win to individuals and  teams. So remember from our gradient descent  
Doing a gradient descent step
from scratch that the one bit we don't want to do  from scratch is calculating derivatives because  
it's just manual and boring so, to get Pytorch to  do it for us you have to say: well, what things do  
you want derivatives for and, of course, we want  it for the coefficients. So then we have to say   requires_grad_(). And remember, very important  in Pytorch, if there's an underscore at the end,  
that's an in-place operation, so this is actually  going to change coeffs. It also returns them,  
right?, but it also changes them in place. So now  we've got exactly the same numbers as before but   with requires_grad_ turned on. So now when we  calculate our loss, that doesn't do any other  
calculations, but what it does store is a gradient  function, it's the function that Python has  
remembered that it would have to do to undo  those steps to get back to the gradient   and to say oh please actually call that  backward gradient function you call backward.
And at that point it sticks into a .grad  attribute the coefficient… the coefficients  
gradients, so this tells us that if  we increased the ‘Age’ coefficient,  
the loss would go down, so  therefore we should do that, right?
So since negative means increasing this would  decrease the loss that means we need to –if you  
remember back to the gradient descent from scratch  notebook– we need to subtract the coefficients  
times the learning rate. So we haven't got any   particular ideas yet of how to set the  learning rate so for now I just pick,  
just try a few, and still find out what works  best, in this case I found 0.1 worked pretty well.
So I now subtract –so again this is sub_, so  subtract in place– from the coefficients their  
gradient times the learning rate; and so the loss  has gone down, that's great, from 0.54 to 0.52.  
So there is one step. So we've now got  everything we need to train a linear model…
So let's do it. Now, as we discussed  last week, to see whether your model  
is any good it's important that you split  your data into training and validation.  
Training the linear model
For the Titanic data set it's actually  pretty much fine to use a random split   because, back when my friend Markimarket and  I actually created this competition for Kaggle  
many years ago, that's basically what we did –if I  remember correctly–. So we can split them randomly  
into a training set and a validation set. So  we're just going to use fast.ai for that. There's,  
you know, it's very easy to do it manually  with Numpy or Pytorch, you can use scikit-learn  
train_test_split(). I'm using fast.ai's here,  partly because it's easy just to remember one  
way to do things and this works everywhere, and  partly because in the next notebook we're going   to be seeing how to do more stuff in fast.ai so i  want to make sure we have exactly the same split.
So those are…
a list of the indexes of the rows that will  be, for example, in the validation set,  
that's why I call it validation split. So to  create the validation independent variables you   have to use those to index into the independent  variables and ditto for the dependent variables.  
And so now we've got… our independent variable training set and our  
validation set, and we've also got  the same for the dependent variables.
So, like I said before, I normally take stuff  that I've already done in a notebook, seems to be  
working, and put them into functions; so here's  the step which actually updates coefficients,  
so let's chuck that into a function. And then  the steps that go: calc_loss(), loss.backward(),  
update coefficients (update_coeffs()) and then  print the loss, we chuck that in one function;   so just copying and pasting stuff into cells here.  And then the bit on the very top of the previous  
section that got the round of numbers minus 0.5  requires_grad_()... chuck that in the function.  
So here we've got something that initializes  coefficients, something that does one epoch   by updating coefficients, so we can put that into,  them together into something that trains the model  
for n epochs, with some learning  rate, by setting the manual seed,   initializing the coefficients, doing one epoch  in a loop and then return the coefficients.
So let's go ahead and run that function.   So it's printing at the end of each one the loss.  And you can see the loss going down from 0.53  
down, down, down, down, down to a bit under 0.3.  So that's good, we have successfully built and  
trained a linear model on a real data set –I mean  it's a Kaggle data set– but it's important to like  
not underestimate how real Kaggle data sets are,  they're real data. And this one's a playground  
data set so it's not like anybody actually cares  about predicting who survived the titanic because   we already know, but it has all the same  features of, you know, different data types  
and missing values and normalization and so forth.  So, you know, it's a good, it's a good playground.
So, it'd be nice to see what the  coefficients are attached to each variable,   so if we just zip together the independent  variables and the coefficients –and  
we don't need the requires_grad_()  anymore– and create a dict of that…
there we go, so it looks like older  people had less chance of surviving  
–that makes sense–. Males  had less chance of surviving   –also makes sense–. So it's good to kind of  eyeball these and check that they seem reasonable.
Now, the metric for this Kaggle competition is  not mean absolute error. It's accuracy. Now, of  
Measuring accuracy
course, we can't use accuracy as a loss function,  because it doesn't have a sensible gradient,   really. But we should measure accuracy to see  how we're doing, because that's going to tell us  
how we're going to get the thing that the  Kaggle competition cares about. So we can   calculate our predictions, and we'll just say:  okay, well any time the prediction's over 0.5,  
we'll say that's predicting survival. So that's  our - predictive of survival. This is the actual  
in a validation set. So if they're the same,  then we predicted it correctly. So here's  
“are we right or wrong” for the first 16 rows.  We're right more often than not. So if we take  
the mean of those - remember true equals one -  then that's our accuracy. So we're right about  
79% of the time. So it's not bad. Okay so  we've successfully created something that's  
actually predicting who survived the  Titanic. That's cool - from scratch.  
So let's create a function for that - an  accuracy function - that just does what I showed,  
and there it is. Yeah, I say, another thing like, you  know, my weird coding thing for me,  
you know - weird as in not that common -  is I use less comments than most people,  
because all of my code lives in notebooks. And,  of course, in the real version of this notebook…  
is full of prose. Right? So when I've  taken people through a whole journey  
about what I've built here and why I've built  it and what intermediate results are and check   them along the way, the function itself in my, you  know, for me, doesn't need extensive comments. And  
I'd rather explain the thinking of how I got there  and show examples of how to use it and so forth.
Okay! Now… Here's the first few predictions we made…
Using sigmoid
And, some of the time, we're predicting negatives  for survival and greater than one for survival  
which doesn't really make much sense. Right?  People either survived - one - or they   didn't - zero. It would be nice if we had a way to  automatically squish everything between zero and  
one. That's gonna make it much easier to optimize.  The optimizer doesn't have to try hard to hit  
exactly one or hit exactly zero, but it can just  like try to create a really big number to mean  
“survive”, to a really small number to  mean “perished”. Here's a great function.
Here's a function that as I increase  - let's make it even bigger range -  
as my numbers get beyond four or five, it's  asymptoting to one. And on the negative side,  
as they get beyond negative four or five,  they asymptote to zero - or to zoom in a bit.
But then, around about zero, it's  pretty much a lot a straight line.   This is actually perfect. This is exactly  what we want. So, here is the equation:  
1 over 1 plus a to the negative minus x.  And this is called the “sigmoid function”.  
By the way, if you haven't checked out  sympy before, definitely do so. This  
is the symbolic Python package which can do…  It's kind of like Mathematica or Wolfram style  
symbolic calculations including the ability to  plot symbolic expressions, which is pretty nice.  
Pytorch already has a sigmoid function. I  mean it just calculates this, but it does it   in a more optimized way. So, what if we replaced  calc_preds()? Remember, before calc_preds() was  
just this. What if we took that  and then put it through a sigmoid?  
So calc_preds() will now… Basically, the bigger  - oops - the bigger this number is the closer  
it's going to get to one, and the smaller  it is the closer it's going to get to zero. This should be a much easier thing to optimize and  ensures that all of our values are in a sensible  
range. Now, here's another cool thing about using  Jupyter plus Python. Python is a dynamic language.  
Even if though I called calc_preds(),  train_model() calls one_epoch()  
which calls calc_loss() which calls calc_preds(),  I can redefine calc_preds() now, and I don't  
have to do anything - that's now inserted into  Python's symbol table and that's the calc_preds()  
that train_model() will eventually call. So if I  now call train_model(), that's actually going to  
call my new version of calc_preds().  So it's a really neat way of   doing exploratory programming in  Python. I wouldn't, you know, release,  
you know, a library that redefines calc_preds()  multiple times. You know? When I'm done,   I would just keep the final version of course.  But it's a great way to try things, as you'll see.
And so, look what's happened. I found I was able  to increase the learning rate from 0.1 to 2.   It was much easier to optimize as I guessed.  And the loss has improved from 0.295  
to 0.197. The accuracy has improved  from 0.79 to 0.82, nearly 0.83.  
So, as a rule, this is something that we're  pretty much always going to do when we have a  
binary dependent variable - dependent variable  that's one or zero - is the very last step is  
chuck it through a sigmoid. Generally speaking,  if you're wondering why is my model with a binary  
dependent variable not training very well, this is  the thing you want to check: “Oh are you chucking  
it through a sigmoid or is the thing you're  calling chucking it through a sigmoid or not?”  
It can be surprisingly hard to find out if that's  happening. So, for example, with HuggingFace   transformers. I actually found I had to look in  their source code to find out and I discovered  
that something i was doing wasn't and didn't seem  to be documented anywhere. But it is important  
to find these things out. As we'll discuss  in the next lesson, we'll talk a lot about  
neural net architecture details, but the  details we'll focus on are what happens  
to the inputs at the very first stage and what  happens to the outputs at the very last stage.   We'll talk a bit about what happens in the  middle but a lot less. And the reason why is it's  
the things that you put into the inputs that's  going to change for every single data set you do  
and what do you want to happen to the outputs  which is going to happen for every different   target that you're trying to hit. So those are the  things that you actually need to know about. So,  
for example, this thing of, like, well, you need  to know about the sigmoid function and you need to   know that you need to use it. fast.ai is very good  at handling this for you. That's why we haven't  
had to talk about it much until now. If you say,  “Oh, it's a category block dependent variable, you  
know, it's going to use the right kind of thing  for you. But most things are not… so convenient.
John, is there a question? Yes, there is. It's back in the sort  of the feature engineering topic,  
but a couple of people have liked it so I  thought we'd put it out there. So, Shivam says,  
“One concern I have while using get_dummies” -  right so it's in that get_dummies phase - “‘is  
what happens while using test data. I have a new  category - let's say male, female, and other - and  
this will have an extra column missing from the  training data. How do you take care of that?”
That's a great question. Yeah, so…  
normally you've got to think about this pretty  carefully and check pretty carefully, unless you   use fast.ai. So fast.ai always creates an extra  category called other. And at test time… inference  
time… if you have some level that didn't exist  before, we put it into the other category for you.  
Otherwise, you basically have to do that  yourself… or at least check, you know.
Generally speaking, it's pretty likely that  otherwise your extra level will be silently  
ignored - you know? - because it's going to  be in the data set, but it's not going to   be matched to a column. So yeah. It's a good point  and definitely worth checking. For categorical  
variables with lots of levels, I actually normally  like to put the less common ones into another  
category, and, again, that's something  that fast.ai will do for you automatically.
But, yeah, definitely something to  keep an eye out for. Good question!
Okay, so before we take our break, we'll just  do one last thing, which is we will submit this   to Kaggle, because I think it's quite cool that  we have successfully built a model from scratch.  
Submitting to Kaggle
So Kaggle provides us with a test.csv, which is  exactly the same structure as the training.csv,  
except that it doesn't have a “Survived”  column. Now interestingly, when I tried to   submit to Kaggle, I got an error in my  code saying that “oh, one of my Fares  
is empty”. So that was interesting, because  the training set doesn't have any empty Fares.  
So sometimes this will happen that the…  the training set and the test set have   different things to deal with. So in this  case, I just said “oh, there's only one row,  
I don't care”. So I just replaced the  empty one with the… with a zero for Fare.
So then I just copied and pasted the  pre-propressing… the pre-processing steps from  
my training dataframe, and stuck them here for  the test data frame and the normalization as well.  
And so now I just call calc_preds,  is it greater than 0.5,  
turn it into a zero or one,  because that's what Kaggle expects,   and put that into the Survived column,  which previously remember, didn't exist.  
So then finally I create a dataframe with  just the two columns, ID and Survived…
Stick it in a csv file, and then I can call the  Unix command head, just to look at the first few  
rows. And if you look at the Kaggle competition's  data page, you'll see this is what the submission  
file is expected to look like. So that made me  feel good. So I went ahead and submitted it.  
I didn't mention it… okay so anyway, I submitted  it and I remember I got like, I… I think I was,  
basically right in the middle, about 50%, you  know, better than half the people who have entered   the competition, worse than half the people. So  you know, solid, middle of the pack result for a  
linear model from scratch. I think it's a  pretty good result. So it's a great place   to start. So let's take a 10 minute break. We'll  come back at 07:17 and continue on our journey.
All right, welcome back. Um…
Using matrix product
You might remember from  Excel that after we did the
some product version, we then  replaced it with a matrix-multiply
uh wait, not there, must be here, here we are.  
Where the matrix-multiply. So  let's do that step now. So…
Um, “matrix times vector dot sum over axis  equals 1”, is the same thing as matrix-multiply.  
So here is the “times dot sum” version. Now we  can't use this character for a matrix-multiply  
because it means element-wise operation. All  of the “times”, “plus”, “minus”, “divide”  
in Pytorch, Numpy mean element-wise. So,  corresponding elements. So in Python instead we  
use this character. As far as I know it's pretty  arbitrary. It's one of the ones that wasn't used.  
So that is an official Python... it's  a bit unusual, it's an official Python   operator. It means matrix-multiply, but Python  doesn't come with an implementation of it. So  
because we've imported, because these are  tensors, and in Pytorch, that will use   Pytorch’s. And as you can see, they're exactly  the same. So we can now just simplify a little  
bit what we had before. calc_preds is  now torch.sigmoid of the matrix-multiply.  
Now there is one thing I'd like to move towards  now, is that we're going to try to create   a neural net in a moment. And so that means rather  than treat this as a “matrix times a vector”,  
I want to treat this as a “matrix times a  matrix”, because we're about to add some more  
columns of coefficients. So we're going to change  indeps@coeffs, so that rather than creating  
an n_coeff vector, we're going to  create an “n_coeff by 1” matrix.  
So in math, we would probably call that  a column vector, but I think that's a   kind of a dumb name in some ways, because it's  it's a matrix, right? It's a rank-two tensor. Um.
So, like the matrix multiplier will work fine  either way, but the key difference is that,  
if we do it this way, then the result  of the matrix multiplier will also be  
a matrix. It'll be again a “n rows by one”  matrix. That means when we compare it to the  
dependent variable, we need the dependent  variable to be an “n rows by one” matrix   as well. So effectively we need to take the  n-rows-long vector, and turn it into an “n rows  
by one” matrix. So there's some useful… very  useful, and at first, maybe a bit weird, notation  
in Pytorch, Numpy for this. Which is if I  take my training dependent variables vector,  
I index into it, and colon means  every row, right? So in other words,
that just means the whole vector, right?  It's the same, basically, as that.
And then I index into a second dimension.  Now this doesn't have a second dimension.  
So there's a special thing you can do, which  is, if you index into a second dimension   with a special value “None”, it  creates that dimension. So this…
has the effect of adding an extra, trailing  dimension to train_dependent(trn_dep). So  
it turns it from a vector to a matrix with one  column. So if we look at the shape after that…
as you see, it's now got,   we call this a unit axis. It's got a  trailing unit axis. 713 rows and one column.
So now if we train our model, we'll  get coefficients just like before.
Except that
it's now a column vector, also known as a  rank-two matrix with a trailing unit axis.  
Okay, so that hasn't changed anything. It's just  repeated what we did in the previous section,   but it's kind of set us up to expand, because  now that we've done this using matrix-multiply,  
we can go crazy, and we can go ahead and create  a neural network. So with our neural network,  
A neural network
remember, back to the Excel days, notice  here, it's the same thing, right? We created  
a column vector, but we didn't create a  column vector. We actually created a matrix  
with, kind of, two sets of coefficients. So when  we did our matrix-multiply, every row gave us  
two sets of outputs, which  we then chuck through Relu,  
right, which, remember, we just used an  if-statement and we added them together. So…
our coeffs now to make a proper neural net. We  need one set of coeffs here, and so here they are,  
torch.rand(), n_coeffs by what? Well, in Excel,  we just did two, because I kind of got bored of  
getting everything working properly. But you don't  have to worry about feeling right and creating  
columns, and blah blah blah and… In Pytorch,  you can create as many as you like. So I made it  
something you can change. I call it n_hidden,  number of hidden activations. I just set it to 20.  
And as before, we centralize them by making  them go from -0.5 to 0.5. Now when you do  
stuff by hand, everything does get more fiddly.  If our coefficients aren't… if they're too big  
or too small, it's not going to train it at  all. Basically, the gradients will, kind of,  
vaguely point in the right direction, but you'll  jump too far or not far enough or whatever.   So I want my gradients to be about the  same as they were before. So I divide by  
n_hidden because otherwise, at the next step  when I add up the… the next matrix-multiply,  
it's going to be much bigger than it was before.  So it's all very fiddly. So then I want to take…  
so that's going to give me, for every row it's  going to give me 20 activations. 20 values,  
right? Just like in Excel we had two values  because we had two sets of coefficients.  
And so to create a neural net, I now  need to multiply each of those 20 things…
by a coefficient. And this time it's going to be a column vector,  because I want to create one output predictor  
of survival. So again, torch.rand(), and this time  the n_hidden will be the number of coefficients by  
one. And again, like trying to find something that  actually trains properly required me some fiddling  
around, to figure out how much to subtract and I  found if I subtract 0.3, I could get it to train.  
And then finally, I didn't need a constant term  for the first layer as we discussed, because our  
dummy variables have, you know, “n” columns  rather than “n minus one” columns, but layer two,  
absolutely needs a constant term, okay? And  we could do that as we discussed last time   by having a column of ones, although in practice  I actually find it's just easier just to create  
a constant term. Okay, so here is  a single, scalar random number.  
So those are the coefficients we need. One set  of coefficients to go from input to hidden,   one goes from hidden to a single output and a  constant. So they're all going to need grad.  
And so now we can change how  we calculate predictions.   So we're going to pass in all of our coefficients.  So a nice thing in Python is, if you've got a list  
or a tuple of values, on the left hand side you  can expand them out into variables. So this is  
going to be a list of three things. So we'll call  them L1, layer 1, layer 2, and the constant term,  
because those are the list of three things we  returned. So in Python, if you just chuck things  
with commas between them like this, it creates a  tuple. A tuple is a list. It's an immutable list.  
So now we're going to grab those three things.  So step one is to do our matrix multiply.  
And as we discussed, we then have to replace  the negatives with zeros, and then we put that  
through our second matrix-multiply. So our second  layer, and add the constant term. And remember,  
of course at the end, chuck it through a  Sigmoid(). So here is a neural network.
Now update_coeffs() previously  subtracted the coefficients…   the gradients times the learning rate  from the coefficients. But now we've got  
three sets of those, so we have  to just chuck that in a for loop.
So change that as well. And now we can go ahead  and train our model. Ta-da! We just trained a  
model and how does that compare? So the loss  function's a little better than before. Accuracy?
Exactly the same as before. And, you  know, I will say it was very annoying  
to get to this point, trying to get these  constants right, and find a learning rate that   worked. Like, it was super fiddly. But you know,  we got there. We got there. It's a very small test  
set. I don't know if this is necessarily better  or worse than the linear model, but it's certainly   fine. And I think that's pretty cool that we  were able to build a neural net from scratch…
that's doing pretty well. But I hear that all  the cool kids nowadays are doing deep learning,  
Deep learning
not just neural nets. So we better make this deep  learning. So, this one only has one hidden layer.  
So, let's create one with n hidden layers. So,  for example, let's say we want two hidden layers,  
10 activations in each. You can put as many as  you like here, all right? So init_coeffs() now  
is going to have to create a torch.rand()  for every one of those hidden layers,  
and then another torch.rand()  for your constant terms.
Stick requires_grad() in all of them. And  then we can return that. So that's how we  
can just initialize as many layers as we want, of  coefficients. So the first one… the first layer,  
so the sizes of each one, the first layer  will go from n_coeff to 10. The second matrix  
will go from 10 to 10, and the third matrix  will go from 10 to 1. So it's worth, like,  
working through these matrix multipliers on, like,  a spreadsheet or a piece of paper or something,   to kind of convince yourself that there's a  right number of activations at each point.
And so then we need to update  calc_preds(), so that rather   than doing each of these steps manually,  we now need to loop through all the layers.
Do the matrix-multiply, add the constant, and as  long as it's not the last layer, do the Relu().  
Why not the last layer? Because remember, the last  layer has Sigmoid(). So these things about, like…  
remember what happens on the last layer? This is  an important thing you need to know about. You   need to kind of check if things aren't working.  What's your… this thing here, is called the  
activation function torch.sigmoid(), and F.relu().  They're the activation functions for these layers.  
One of the most common mistakes amongst people  trying to kind of create their own architectures,  
or kind of variants of architectures, is to mess  up their final activation function, and that makes  
things very hard to train. So make sure we've  got a torch.sigmoid() at the end, and no Relu()  
at the end .So there's our deep learning  cal_preds(). And then just one last change,  
is now when we update our coefficients, we go  through all the layers and all the constants.
And again, there was so much messing  around here, with trying to find, like,   exact ranges of random numbers. That end up  training okay, but eventually I found some,  
and as you can see, it gets to about the  same loss and about the same accuracy.
Linear model final thoughts
This code is worth spending time with.  And when the code is inside a function,  
it can be a little difficult to experiment with.  So you know, what I would be inclined to do,   to understand this code, is to kind of copy  and paste this cell, make it so it's not in  
a function anymore. And then use Ctrl Shift  Dash to separate these out into separate cells,  
right? And then try to kind of set it up so you  can run a single layer at a time, or a single   coefficient. Like, make sure you can see what's  going on, okay? And that's why we use notebooks.  
It's so that we can experiment, and it's only  through experimenting like that, that at least   for me, I find that I can really understand  what's going on. Nobody can look at this  
code and immediately say, I don't think anybody  can, “I get it, that all makes perfect sense.”,  
but once you try running through it yourself,  you'll be like “oh! I see why that's as it is!”.
So you know one thing to  point out here is that our…
neural nets and deep learning models  didn't particularly seem to help.  
So does that mean that deep  learning is a waste of time, and you   just did five lessons that you shouldn't have  done? No, not necessarily. This is a playground  
competition. We're doing it because it's easy  to get your head around. But for very small data  
sets like this, with very very few columns,  and the columns are really simple, you know,  
deep learning is not necessarily going to give  you the best result. In fact, as I mentioned, um…
Nothing we do is going to be  as good as a carefully designed  
model that uses just the name column.
So you know, I think that's an  interesting insight, right, is that   the kind of data types which have a very  consistent structure, like, for example,  
images or natural language text  documents, quite often you can  
somewhat brainlessly chuck a deep learning  neural net at it, and get a great result.  
Generally, for tabular data, I find  that's not the case. I find I normally   have to think pretty long and hard about  the feature engineering in order to get  
good results. But once you've got good features,  you then want a good model, and so you… you know…
and generally, like, the more features you have,  and the more levels in your categorical features,   and stuff like that, you know, the more value  you'll get from more sophisticated models.  
But yeah, I definitely would say a… an insight  here is that, you know, you want to include simple  
baselines as well. And we're going to be seeing  even more of that in a couple of notebooks time.
Why you should use a framework
So we've just seen how you can build stuff from  scratch. We now say why you shouldn't. I mean I  
say you shouldn't. You… you should learn, but  why you probably won't want to in real life.  
When you're doing stuff in real life, you  don't want to be fiddling around with all this  
annoying initialization stuff, and learning  rate stuff, and dummy variable stuff,  
and normalization stuff, and so forth. Because we  can do it for you. And it's not like everything's  
so automated that you don't get to make choices.  But you want, like, you want to make the choice   not to do things the obvious way, and have  everything else done the obvious way for you. So  
that's why we're going to look at this - “Why you  should use… use a framework” notebook. And again,   I'm going to look at the clean version of it. And  again, in the clean version of it, step one is to  
download the data as appropriate for  the Kaggle or non-Kaggle environment,   and set the display options, and set the  random seed, and read the dataframe. All right,  
Prep the data
now, there was so much fussing around with the  “doing it from scratch” version that I did not  
want to do any feature engineering, because every  column I added was another thing I had to think   about - dummy variables and normalization  and random coefficient initialization  
and blah blah blah. But with a framework,  everything's so easy, you can do all the  
feature engineering you want. Because this isn't  a lesson about feature engineering, instead,  
I plagiarized entirely from this fantastic  advanced feature engineering tutorial on Kaggle.
And what this tutorial found was that, in  addition to the “LogFare” we've already done,  
that you can do cool stuff with the “Deck”,  with adding up the number of family members,   whether people are traveling alone, how many  people are on each ticket. And finally, we're  
going to do stuff with the name, which is we're  going to grab the Mr, Miss, Mrs, Master, whatever.
So we're going to create a function to, like,  do some feature engineering. And if you want to   learn a bit of Python, Pandas, here's some great  lines of code to step through one by one. And  
again, like, take this out of a function, put them  into individual cells, run each one, look up the  
tutorials, what does str() do, what does map()  do, what does groupby() and transform() do,  
what does value_counts() do? Like, these are  all, like, part of the reason I put this here   was for folks that haven't done  much, if any, Pandas, to have some,  
you know, examples of functions that I  think are useful. And I actually refactored   this code quite a bit, to try to show off some  features of Pandas I think are really nice.  
So we'll do the same random split as  before, so passing in the same seed.
And so now we're going to do the same set  of steps that we did manually with fastai.  
So we want to create a tabular model dataset  based on a Pandas dataframe. And here is the  
dataframe. These are the train versus  validation splits I want to use.
Here's a list of all the stuff I want done. Please  deal with dummy variables for me, deal with…  
deal with missing values for me,  normalize continuous variables for me.  
I'm going to tell you which ones are the  categorical variables. So here's, for example,   “Pclass” was a number, but I'm telling fastai to  treat it as categorical. Here's all the continuous  
variables. Here's my dependent variable,  and the dependent variable is a category.
So create dataloaders from that place, and  save models right here in this directory.
That's it! That's all the pre-processing I need to  do, even with all those extra engineered features.  
Train the model
Create a learner… okay, so this, remember, is  something that contains a model  
and data, and I want you to put in two hidden  layers with 10 units and 10 units, just like  
we did in our final example. What learning rate  should I use? Make a suggestion for me, please. So  
call lr_find(). You can use this for any fastai  model. Now what this does, is, it starts at a  
learning rate that's very very small, 10 to the  negative seven, and it puts in one batch of data,  
and it calculates the loss. And then it puts  through… and then it increases the learning   rate slightly, and puts through another batch  of data. And it keeps doing that for higher and  
higher learning rates. And it keeps track of the  loss as it increases the learning rate. Just one   batch of data at a time. And what happens is for  the very small learning rates, nothing happens.  
But then once you get high enough, the loss  starts improving, and then as it gets higher, it  
improves faster until you make the learning rate  so big that it overshoots, and then it kills it.  
And so, generally, somewhere around  here, is the learning rate you want.  
fastai has a few different ways of recommending  a learning rate. You can look up the docs to see   what they mean. I generally find if you choose  slide and valley, and pick one between the two,  
you get a pretty good learning rate. So here we've  got about .01, and about 0.08, so I picked 0.03.  
So just run a bunch of epochs. Away it goes. Tada!
This is a bit crazy. After all that we've ended  up exactly the same accuracy as the last two   models. That's just a coincidence, right, I mean  there's nothing particularly about that accuracy.  
Aand so, at this point we can now submit that  to Kaggle. Now, remember with the linear model,  
Submit to Kaggle
we had to repeat all of the pre-processing  steps on the test set in exactly the same way.  
Don't have to worry about  it with fastai. In fastai,   I mean, we still have to deal with  the fill-missing (fillna) for Fare,  
because that's… that's that. We have to  add our feature engineering features,   but all the pre-processing, we just have to use  this one function called test_dl(). And that says  
create a dataloader that contains exactly the  same pre-processing steps that our learner used.  
And that's it. That's all you need. So just  because you… you want to make sure that your   inference-time transformations, pre-processing  are exactly the same as a training time.  
So this is the magic method which does that.  Just one line of code. And then to get your  
predictions, you just say get_preds()  and pass in that dataloader I just built.  
And so then these three lines of  code are the same as the previous   notebook. And we can take a look at  the top and you can see… there it is.
So how did that go?
I don't remember… no, i didn't say…   I… I think it was again basically middle  of the pack, if I remember correctly.
So one of the nice things about,   now that it's so easy to, like, add features  and build models, is we can experiment  
with things much more quickly. So I'm going  to show you how easy it is to experiment with,  
you know, what's often considered a fairly  advanced idea, which is called ensembling.  
Ensembling
There's lots of ways of doing ensembling,  but basically ensembling is about creating   multiple models, and combining their predictions.  And the easiest kind of ensemble to do,  
is just to, literally, just build multiple  models. And so each one is going to have  
a different set of randomly initialized  coefficients, and therefore each one is going   to end up with a different set of predictions.  So I just create a function called ensemble(),  
which creates a learner, exactly the same  as before, fits exactly the same as before,  
and returns the predictions. And so we'll just  use a list comprehension to do that five times.  
So that's going to create  a set of five predictions.
Done. So now we can take all those predictions  and stack them together, and take the mean  
over the rows. So that's going to give  us the… what's actually… sorry, the mean   over… the over the first dimension. So  the mean over the sets of predictions.  
And so that will give us the average prediction of  our five models. And again we can turn that into  
a csv and submit it to Kaggle. And that one…  I think that went a bit better. Let's check.
Yeah, okay. So that one, actually finally gets  into the top 20… 25 in the competition. So I mean,  
not amazing by any means, but you can  see that, you know, this simple step of  
creating five independently trained models,  just starting from different starting points,   in terms of random coefficients, actually  improved us from top 50 to top 25. John?
Framework final thoughts
Is there an argument, because you've got a  categorical result, zero one effectively, is there   an argument that you might use the mode of the  ensemble, rather than the numerical mean? I mean,
yes, there's an argument that's been made,   and yeah, that's something I would just  try. I generally find it's less good,  
but not always. And I don't feel like I've got  a great intuition as to why, and I don't feel  
like I've seen any studies as to why. You could  predict, like, there's a few… there's… there's  
at least three things you could do, right? You  could take the “is it greater or less than 0.5”,
ones and zeros, and average them. Or you could  take the mode of them. Or you could take the   actual probable… probability predictions, and take  the average of those, and then threshold that.  
And I've seen examples where, certainly,  both of the different averaging versions,   each of them has been better. I don't think  I've seen one where the mode's better, but  
that was very popular back in the 90s. So…
yeah so be it is so easy to try,  you may as well give it a go.
Okay. We don't have time to finish the next  notebook but let's make a start on it.
How random forests really work
So the next notebook is random forests.  How random forests really work.  
who here has heard of random forests before?  Nearly everybody. Okay. So very popular.  
Developed I think initially in 1999, but you  know gradually improved in popularity during the  
2000s. I was like, everybody kind of knew me as Mr  Random Forests for… for years. I implemented them,  
like, a couple of days after the original  technical report came out. I was such a fan. All  
of my early Kaggle results from random forests.  I love them, and I think hopefully you'll see why  
I'm such a fan of them, because they're so  elegant, and they're almost impossible to mess up.  
A lot of people will say ,like, “oh, why are  you using machine learning? Why don't you use   something simple, like logistic regression?”.  And I think, like, oh gosh, in industry  
I've seen far more examples of people screwing  up logistic regression than successfully using   logistic regression, because it's very, very,  very, very difficult to do correctly. You know,  
you've got to make sure you've got the correct  transformations, and the correct interactions,   and the correct outlier handling, and blah  blah blah. And anything you get wrong,  
the entire thing falls apart. Random forests, I…  it's very rare to… that I've seen somebody screw  
up a random forest in industry. They're  very hard to screw up, because they're…   they're so resilient, and you'll  see why. So in this notebook…
just, by the way, rather than importing Numpy  and Pandas and Matplotlib and blah blah blah,   there's a little handy shortcut which is, if  you just import everything from fastai.imports,  
that imports all the things that you normally  want. So, I mean, it doesn't do anything special,  
it's just save some messing around. So again,  we've got our cell here to grab the data.
Data preprocessing
And I'm just going to do some  basic pre-processing here
with my fillna() for the Fare, only  needed for the test set, of course.  
Grab the modes and do the fillina()  on the modes. Take the logFare.  
And then I've got a couple of new steps  here, which is converting Embarked and Sex  
into categorical variables. What does that  mean? Well, let's just run this on both the   dataframe… dataframe and the test dataframe. Split  things into set… into categories and continuous.
And Sex is a categorical  variable. So let's look at it.  
Well that's interesting. It looks exactly the same  as before. Male and Female. But now it's called   a category, and it's got a list of categories.  What's happened here, well, what's happened is  
Pandas has made a list of all of  the unique values of this field.   And behind the scenes, if you look at the  cat.codes, you can see behind the scenes,  
it's actually turned them into numbers.  It looks up this One into this list,  
to get Male. Looks at this Zero into this  list, to get Female. So when you print it out,   it prints out the friendly version,  but it stores it as numbers. Now,  
you'll see in a moment why this is helpful,  but a key thing to point out is, we're not  
going to have to create any dummy variables.  And even that first, second or third class,  
we're not going to consider that categorical  at all. And you'll see why in a moment.
A random forest is an ensemble of trees.  
Binary splits
A tree is an ensemble of binary splits, and  so we're going to work from the bottom up.   We're going to first work… we're going to  first learn about, what is a binary split.
And we're going to do it by  looking at example. Let's consider,   what would happen if we took all  the passengers on the Titanic,  
and group them into males and females. And let's  look at two things. The first is, let's look at  
their Survival rate. So about 20 Survival rate  for males, and about 75 for females. And let's  
look at the histogram. How many of them are there?  About twice as many males as females. Consider  
what would happen if you created the world's  simplest model, which was… what sex are they.
That wouldn't be bad, would it? Because there's a  big difference between the males and the females,  
a huge difference in Survival rate. So if we  said, oh, if you're a man, you probably died;  
If you're a woman, you probably survived or… not  just a man or a boy, so male or female. That would  
be a pretty good model, because it's done a good  job of splitting it into two groups that have very   different survival rates. This is called a binary  split. A binary split is something that splits  
the rows into two groups. Hence, binary. Let's  talk about another example of a binary split.  
I'm getting ahead of myself. Before we do that,  let's look at what would happen if we used this  
model. So if we created a model which just looked  at Sex, how good would it be? So, to figure that  
out, we first have to split into training  and test sets. So let's go ahead and do that.
And then let's convert all of our  categorical variables into their codes.  
So we've now got zero, one, two, whatever.  We don't have Male or Female there anymore.  
And let's also create something that returns the  
independent variables, which we'll call the “xs”,  and the dependent variable, which we'll call y.  
And so we can now get the xs and the y for each  of the training set and the validation set.   And so now let's create some predictions.  We'll predict that they survived  
if their Sex is zero. So if they're female. So  how good is that model? Remember I told you that  
to calculate mean absolute error, we can  get Scikit-learn, or Pytorch, whatever,   do it for us instead of doing it ourselves. So  just showing you, here's how you do it just by  
importing it directly. This is exactly  the same as the one we did manually   in the last notebook. So that's a 21.5 percent  error. So that's a pretty good model. Um…
Could we do better? Um… well here's another example. What about  Fare. so Fare is different to Sex, because Fare  
is continuous. Or LogFare I think. But we could  still split it into two groups. So here's… for  
all the people that didn't survive, this is their  median Fare here, and then this is their quartiles  
for big Fare, and quartiles for small Fare. And  here's the median Fare for those that survived  
and their quartiles. So you can see the  median Fare for those that survived is higher   than the median fare for those that didn't.  We can't create a histogram exactly for  
Fare because it's continuous. We could bucket it  into groups to create a histogram. So I guess we  
can create a histogram. That wasn't true. What I  should say is, we could create something better,  
which is a kernel density plot,  which is just like a histogram,   but it's like with infinitely small bins. So we  can see most people have a LogFare of about two.
So what if we split on about, a bit under three.  
You know, that seems to be a point at which  there's a difference in survival, between people  
that are greater than or less than that amount.  So here's another model LogFare greater than 2.7
Oh, much worse. 0.336 versus 0.215. Well, I  don't know, maybe there's something better.
We could create a little interactive tool. So…
What I want is something that can give us a  quick score of how good a binary split is.  
And I want it to be able to work, regardless  of whether we're dealing with categorical,   or continuous, or whatever data. So  I just came up with a simple little  
way of scoring, which is I said, okay,  if you split your data into two groups,  
a good split would be one in which all of the  values of the dependent variable on one side are  
all pretty much the same, and all the dependent  variables on the other side are all pretty much   the same. For example, if pretty much all the  males had the same survival outcome, which is  
“didn't survive”, and all the females had about  the same survival outcome. which is “they did   survive”, that would be a good split, right?  It doesn't just work for categorical variables.  
It would work if your dependent variable  was continuous as well. You basically want  
each of your groups, within group to be as  similar as possible on the dependent variable,  
and then the other group, you want them to be as  similar as possible on the dependent variable.   So how similar is all the things in a group?  
That's a standard deviation. So what I want to  do is, basically, add the standard deviations   of the two groups of the dependent variable. And  then if there's a really small standard deviation  
but it's a really small group, that's not very  interesting. So I'll multiply it by the size,  
right? So this is something which says, what's  the score for one of my groups, one of my sides?  
It's the standard deviation multiplied  by how many things are in that group.   So the total score is the score for the  left-hand side. So all the things in one group.  
Plus the score for the right-hand side, which  is, tilde means not, so “not left-hand side”  
is right-hand side. And then we'll just take the  average of that. So for example, if we split by 6,
is greater than or less than 0.5.That'll  create two groups, males and females,   and that gives us this score. And if we  do LogFare greater than or less than 2.7,  
it gives us this score. And lower score is  better. So Sex is better than LogFare. So  
now that we've got that, we can use  our favorite interact tool to create  
a little GUI. And so we can say, you know, let's  try like, oh, what about this one. Can we… oops…
Can we find something that's a bit better 4.8?  4.5? No, not very good. What about Pclass?
0.468? 0.460? So we can fiddle  around with these. We could do  
the same thing for the categorical  variables. Sorry, we know that Sex,
we can get to 0.407. What about Embarked?
hmm… all right. So looks like Sex might be our  best. Well, that was pretty inefficient, right?  
Would be nice if we could find some automatic  way to do all that. Well, of course we can.  
For example, if we wanted to find what's the best  split point for Age, then we just have to create…
Let's do this again. If we want to find the best split point  for Age, we could just create a list of  
all of the unique values of Age, and try each  one in turn. And see what score we get,.if  
we made a binary split on that level of Age. So  here's a list of all of the possible binary split  
thresholds for Age. Let's go through  all of them for each of them.
Calculate the score and then Numpy  and Pytorch have an argmin() function,  
which tells you what index into that list is the  smallest. So just to show you, here's the scores.
and zero one two three four five six.
oh here, sorry! zero one two three  four five six. So apparently that…
that value has the smallest score.   So that tells us that for Age, the  threshold of 6 would be best. So…
here's something that just calculates that for  a column. It calculates the best split point.
So here's 6.0, right? And it also tells  us what the score is at that point,   which is 0.478. So now we can just go through and
calculates the score for the best split point   for each column. And if we do that,  we find that the lowest score is Sex.
So that is how we calculate the  best binary split. So we now know  
that the model that we created earlier, this one,
is the best single binary split model we can find.   So next week, we're going to be… we're going to  learn how we can recursively do this to create a  
decision tree, and then do that multiple times  to create a random forest. But before we do,  
Final Roundup
I want to point something out, which  is this ridiculously simple thing,   which is, find a single binary split, in short,  is a type of model. It has a name. It's called 1R.  
And the 1R model it turned out, in  a review of machine learning methods   in the 90s, turned out to be one  of the best, if not the best,  
machine learning classifiers for a wide range  of real-world datasets. So that is to say, don't  
assume that you have to go complicated. It's not  a bad idea to always start creating a baseline of  
1R, a decision tree with a single binary  split. And in fact for the Titanic competition,  
that's exactly what we do. If we look at the  Titanic competition on Kaggle, you'll find   that what we did is, our sample submission is  one that just splits into male versus female.  
All right! Thanks, everybody! Hope you found that  interesting and I will see you next lesson. Bye!

---

# 6

OK, so welcome back to lesson 6… not welcome  back to, welcome to lesson 6 — first time we've  
been in lesson 6! Welcome back to Practical Deep  Learning for Coders. We just started looking at  
tabular data last time, and for those of  you who've forgotten what we did was: We  
were looking at the Titanic data set and  we were looking at creating binary splits  
by looking at categorical variables  or binary variables like sex and  
continuous variables, like the  log of the fare that they paid,  
and using those. You know, we also kind of  came up with a score which was basically: How  
good a job did that split do of grouping the  survival characteristics into two groups,  
you know, all of, nearly all of one of  whom survived, nearly all of whom the   other didn't survive so they had like  small standard deviation in each group.
And so then we created the world's simplest  little UI to allow us to fiddle around and   try to find a good binary split. And we did…  we did come up with a very good binary split,  
which was on sex and actually we created  this all automated version. And so this is,  
I think, the first time we can —well not  quite the first time is it?—no, this is,   this is yet another time, I should say, that  we have successfully created a uh, actual  
machine learning algorithm from scratch, this one  is about the world's simplest one, it's “OneR”,  
creating the single rule which does  a good job of splitting your data   set into two parts which differ as much  as possible on the dependent variable.
TwoR model
“OneR” is probably not going to  cut it for a lot of things, though,   it's surprisingly effective but it's uh maybe  we could go a step further. And the other  
step further we could go is we could create  like a “TwoR”, what if we took each of those  
groups: males and females in the Titanic data set  and split each of those into two other groups,  
so split the males into two groups  and split the females into two groups.
So, to do that we can repeat the exact same piece  of code we just did but let's remove sex from it  
and then split the data set into males and females   and run the same piece of code that we  just did before but just for the males.  
And so this is going to be like a “OneR” rule for  how do we predict which males survive the Titanic.
And let's have a look: 38, 37, 38, 38, 38. Okay,  so it's ‘Age’, were they greater than or less than  
six. Turns out to be for the males the biggest  predictor of whether they were going to survive  
that shipwreck. And we can do the same  thing for females, so for females…  
there we go, no great surprise, ‘Pclass’.  So whether they were in first class or not  
was the biggest predictor for females of  whether they would survive the shipwreck.
So that has now given us a decision tree. It is  a series of binary splits which will gradually  
split up our data more and more such that in  the end, in these in the leaf nodes —as we call  
them— we will hopefully get as, you know, much  stronger prediction as possible about survival.
So we could just repeat this step for each  of the four groups we've now created: males,  
kids and older than six. Females, first class  and everybody else, and we can do it again. And  
then we'd have eight groups. We could do that  manually with another couple of lines of code  
or we can just use a decision tree classifier,  which is a class which does exactly that for us.
How to create a decision tree
So there's no magic in here, just  doing what we've just described.   And a decision tree classifier comes  from a library called scikit-learn.  
scikit-learn is a fantastic library  that focuses on, kind of, classical  
non-deep learning ish machine  learning methods like decision trees.
So we can, so to create the exact same  decision tree we can say: please create   a decision tree classifier with at most four  leaf nodes. And one very nice thing it has  
is it can draw the tree for us. So  here's a tiny little draw_tree function.  
And you can see here it's going to first of all  split on sex, now, it looks a bit weird to say sex  
is less than or equal to 0.5 but remember what our  binary characteristics are coded as zero or one,  
so that's just how we, you know,  easy way to say males versus females.  
And then here we've got for the females,  what class are they in, and for the males  
what age are they. And here's our four leaf  nodes. So for the females in first class  
116 of them survived and four of them didn't  so… very good idea to be a “well to do”  
woman on the Titanic. On the  other hand, males adults:  
68 survived 350 died so, very bad idea  to be a male adult on the Titanic.
So you can see you can kind of get a quick summary  of what's going on and one of the reasons people   tend to like decision trees, particularly for  exploratory data analysis, is it does allow us to  
get a quick picture of what are the key  driving variables in this data set and  
how much do they kind of predict  what was happening in the data.
Okay, so it's around the same splits as  us and it's got one additional piece of  
Gini
information we haven't seen before, this  is something called “Gini”. “Gini” is just   another way of measuring how good a split is,  and I've put the code to calculate “Gini” here.  
Here's how you can think of “Gini”: How  likely is it that, if you go into that sample  
and grab one item and then go in again and grab  another item, how likely is it that you're going  
to grab the same item each time. And so, if  that, if the entire leaf node is just people  
who survived or just people who didn't survive the  probability would be one, you get the same time,   same every time. If it was an exactly  equal mix, the probability would be 0.5.  
So that's why we just, yeah, that's where this  formula comes from in the binary case. And in fact  
you can see it here, right, this group here is  pretty much 50-50 so “Gini” is 0.5. Or else this  
grip here is nearly 100% in one class so “Gini”  is nearly zero —so that backwards is one minus.  
And I think I've written it backwards  here as well, so… I've got to fix that.
So this decision tree is, you know, we  would expect it to be more accurate so we  
can calculate its mean absolute error. And for  the “OneR”, so just doing males versus females,  
what was our score? Here we go, 0.407.  
Uh, should we have a, do we have an accuracy  score, somewhere, here we are, 0.336.  
Oh that was for ‘LogFare’ and for ‘Sex’ it was  0.215, okay so 0.215. So that was for the “OneR”  
version, for the decision tree with four leaf  nodes 0.224 so it's actually a little worse,  
right? And I think it just reflects the  fact that this is such a small data set  
and the “OneR” version was so good we  haven't really improved it that much,  
or not enough to really see it amongst the  randomness of such a small validation set.
We could go further to 50, a minimum of  50 samples per Leaf node. So that means  
that in each of these, see how it says “samples”,  which in this case is passengers on the Titanic,  
there's at least... there's 67 people  that were... were female first class  
less than 28. That's how you define that.  So this decision tree keeps building,  
keep splitting until it gets to a point where  there's going to be less than 50 at which point   it stops splitting that... that leaf. So you  can see they're all got at least 50 samples  
and so here's the decision tree that builds. As  you can see, it doesn't have to be like constant  
depth, right? So this group here, which is males  who had cheaper fares and who were older than 20  
but younger than 32... yeah actually  younger than 24. and actually  
super cheap fares and so forth right. So it  keeps going down until we get to that group. So  
let's try that decision tree. That decision  tree has an absolute error of 0.183 so not  
surprisingly you know once we get there it's  starting to look like it's a little bit better.
Making a submission
So there's a model and this is a Kaggle  competition so therefore we should submit  
it to the leaderboard. And. you know.one of  the... you know... biggest mistakes I see  
not just beginners but every level of  practitioner make on Kaggle is not to   submit to the leaderboard, spend months making  some perfect thing right but you actually got  
to see how you're going and you should try and  submit something to the leaderboard every day.  
So you know, regardless of  how rubbish it is because   you want to improve every day. So you want to  keep iterating. So to submit something to the  
leaderboard you generally have to provide a CSV  file and so we're going to create a CSV file.
And we're going to apply the category codes to  get the category for each one in our test set.  
we're going to set the survived  column to our predictions   and then we're going to send that off to a CSV.
And so yeah so I submitted that and I got to score  a little bit worse than most of our linear models  
and neural nets but not terrible, you know.  It was... it's... it's just doing an okay job.
Now one interesting thing for the decision tree  is there was a lot less preprocessing to do,  
did you notice that? We didn't have to create  any dummy variables for... for our categories  
and like, you certainly can create dummy variables  but you often don't have to. So for example  
you know... for… for “class” you know it's... it's  one two or three, you can just split on one, two,  
or three, you know. Even for like, what was that  thing... like the... the “embarkation city code”,  
like we just convert them kind of arbitrarily to  numbers one, two, and three, and you can split on  
those numbers. So with Random Forest also... not  Random Forest, not there yet... decision trees,  
yeah... you can generally get away with not doing  stuff like dummy variables. In fact, even taking  
the log of “fair” we only did that to make our  graph look better but if you think about it,  
splitting on log fare less than 2.7 is exactly the  same as splitting on Fair is less than e to the  
2.7 you know or whatever log base we used, I can't  remember. So all that a decision tree cares about  
is the ordering of the data and this is another  reason that decision tree based approaches are  
fantastic because they don't care at all about  outliers, you know... long tail distributions,  
categorical variables whatever. You can throw  it all in and it'll do a perfectly fine job. So  
for tabular data I would always start by using a  decision tree based approach and kind of create  
some baselines and so forth because it's... it's  really hard to mess it up and that's important.
So yeah... so here for example is “Embarked”,  right? It... it was coded originally as  
the first letter of the city they embarked in,   but we turned it into a categorical variable and  so pandas for us creates this this vocab this list  
of all of the possible values and if you look at  the codes attribute you can see it's that S is  
the... zero one two... so S has become 2, C has  become zero and so forth, right. So that's how  
we're converting the categories, the strings  into numbers that we can sort and group by.
So yeah... so if we wanted to split C into one  group and Q and S in the other we can just do,   okay, less than a quarter one point 0.5. Now  of course if we wanted to split C and S into  
one group and Q into the other we would need two  binary splits first C on one side and Q and S on  
the other and then Q and S into Q versus S and  then the Q and S leaf nodes could get similar  
predictions. So like you do have, like sometimes  it can take a little bit more messing around but  
most of the time I find categorical variables work  fine as numeric in decision tree based approaches  
and as I say here I tend to use dummy variables  only if there's like less than four levels.
Bagging
Now, what if we wanted to make this more  accurate? could we grow the tree further?  
I mean, we could, but, you know, there's only 50  samples in these leaves, right? it's not really…  
you know, if I keep splitting it, the  leaf nodes are going to have so little   data that's not really going to make very useful  predictions. Now there are limitations to how  
accurate a decision tree can be. So what can we  do? We can do something that's actually very,  
I mean, I find it amazing and fascinating,  comes from a guy called Leo Breiman.  
And Leo Breiman came up with… came  up with this idea called bagging.
And here's the basic idea of bagging: let's  say we've got a model that's not very good  
because, let's say, it's a decision tree, it's  really small, we've hardly used any data for it  
right, it's not very good so, it's got  error, it's got errors on predictions.  
It's not a systematically biased error, it's not  always predicting too high or always predicting   too low, I mean, decision trees, you know,  on average will predict the average, right?  
but it has errors. So what I could do  is I could build another decision tree  
in some slightly different way that would have  different splits and it would also be not a  
great model, but it predicts the correct thing on  average, it's not completely hopeless and again,  
you know, some of the errors are a bit  too high and some are a bit too low. And I could keep doing this so, if I could  create building lots and lots of slightly  
different decision trees I'm going to end  up with say a hundred different models,  
all of which are unbiased, all of which are  better than nothing, and all of which have  
some errors a bit high, some bit low, whatever. So  what would happen if I averaged their predictions?  
Assuming that the models are  not correlated with each other   then you're going to end up with errors on  either side of the curve: the correct prediction,  
some are a bit high, some are a bit low,  there'll be this kind of distribution of errors,   right? And the average of those errors will  be zero, and so that means the average of the  
predictions of these multiple uncorrelated models  —each of which is unbiased— will be the correct  
prediction because they have an error of zero. And  this is a mind-blowing insight, it says that if  
we can generate a whole bunch of uncorrelated,  unbiased models, we can average them and get  
something better than any of the individual models  because the average of the error will be zero.
Random forest introduction
So all we need is a way to  generate lots of models.  
Well we already have a great way to build  models —which is to create a decision tree—   how do we create lots of them? how to create  lots of unbiased but different models? Well,  
let's just grab a different subset of the data  each time, let's just grab at random half the rows  
and build a decision tree, and then grab another  half of the rows and build a decision tree,  
and grab another half the rows and build  a decision tree. Each of those decision   trees is going to be not great  —it's only using half the data—  
but it will be unbiased, it will be predicting the  average on average, it will certainly be better   than nothing because it's using, you know, some  real data to try and create a real decision tree.  
They won't be correlated with each other because  they're each random subsets. So that meets all  
of our criteria for bagging. When you do this  you create something called a “Random Forest”.  
Creating a random forest
So let's create one in four lines of code. So  
here is a function to create a decision tree,  so let's say, what —this is just the proportion  
of data— so let's say we put 75% of the data in  each time —or we could change it to 50%— whatever,  
so this is the number of samples in this subset  —I call it n— and so let's at random choose  
n times the proportion we requested from the  sample and build a decision tree from that.  
And so now let's, 100 times, get a tree, and stick  them all in a list using a list comprehension.
And now let's grab the predictions  for each one of those trees  
and then let's stack all those predictions  up together and take their mean.  
And that is a Random Forest. And, what do  we get?, one, two, three, four, five, six,  
seven, eight, that's some, yeah, seven lines  of code. So Random Forests are very simple.  
This is a slight simplification, there's  one other difference that Random Forests   do which is when they build the decision tree  they also randomly select a subset of columns  
and they select a different random subset  of columns each time they do a split.  
And so the idea is you kind of want it to be  as random as possible but also somewhat useful.
So we can do that by creating  a “RandomForestClassifier”,   say how many trees do we want,  how many samples per leaf  
and then fit —that is what we just did—  and here's our mean absolute error which…  
Yeah, again, it's like not as good as  our decision tree but it's still pretty   good and again it's such a small data set  it's hard to tell if that means anything  
and so we can submit that to Kaggle. So  earlier on I created little function to   submit to Kaggle, so now I just create  some predictions and I submit to Kaggle.  
And, yeah, looks like it gave nearly  identical results to a single tree.
Feature importance
Now to one of my favorite things about Random  Forest —and I should say, in… in most real world  
data sets of reasonable size, Random Forest  basically always give you much better results   than decision trees, this is just a small data set  to show you what to do. One of my favorite things  
about Random Forests is we can do something quite  cool with it, what we can do is we can look at the  
underlying decision trees they create,  so we've now got 100 decision trees,  
and we can see what columns did it find a split  on and so, say here, okay well the first thing it  
spit on was ‘Sex’, and it improved the Gini from  0.47 to —now just take the weighted average of  
0.38 and 0.31 weighted by the samples— so that's  probably going to be, I don't know, about 0.33.  
So I'd say, okay, it's like 0.14 Improvement in  Gini thanks to ‘Sex’. And we can do that again:  
okay, we'll then ‘PClass’, you know, how much did  that improve Gini, again, we keep weighting it  
by the number of samples as well. ‘LogFare’, how  much does that improve Gini and we can keep track:  
for each column of, how much in total did  they improve the Gini in this decision tree.  
And then do that for every decision  tree, and then add them up per column  
and that gives you something called a  feature importance plot, and here it is.
And a feature importance plot tells  you how important is each feature,  
how often did the trees pick it and how  much did it improve the Gini when it did.  
And so, we can see from the feature importance  plot that ‘Sex’ was the most important, and  
class was the second most important and  everything else was a long way back.  
And this is another reason, by the way, why a  Random Forest isn't really particularly helpful:   because it's just such a easy split to  do, right? basically all that matters is,  
you know, what class you're in  and whether you're male or female.
And these feature importance plots, remember:  
because they're built on Random Forests,  and Random Forests don't care about, really,  
the distribution of your data and they can handle  categorical variables and stuff like that, that   means that you can basically, any tabular data  set you have, you can just plot this, right away,  
and Random Forests, you know, for most data sets,  only take a few seconds to train, you know, really  
most of a minute or two. And so, if you've got a  big data set and, you know, hundreds of columns:  
do this first and find the 30 columns that  might matter, it's such a helpful thing to do.  
So I've done that, for example, I did some work in  credit scoring, so we're trying to find out which  
things would predict who's going to default on a  loan and I was given something like seven thousand  
columns from the database, now I put it straight  into a Random Forest and found, I think, there was  
about 30 columns that seemed kind of interesting.  I did that like two hours after I started the job  
and I went to the head of marketing  and the head of risk and I told them:  
here's the columns I think that we should  focus on, and they were like: “oh my God,  
we just finished a two-year consulting  project with one of the big consultants,   paid the millions of dollars, and they  came up with a subset of these… «laughs»
Adding trees
There are other things that you can do with  Random Forests along this path, I'll touch  
on them briefly, and specifically I'm going to  look at Chapter 8 of the book, which goes into  
this in a lot more detail. And particularly  interestingly Chapter 8 of the book uses a   much bigger and more interesting data set which  is auction prices of heavy industrial equipment,  
I mean, it's less interesting historically  but more interestingly numerically.
And so, some of the things I  did there on this data set —  
sorry this isn’t from the data set, this  is from the scikit-learn documentation—   they looked at how, as you increase the  number of estimators, so the number of trees,  
how much does the accuracy improve. So I then did  the same thing on our data set, so I actually just  
added up to 40, more and more and more trees,   and you can see that basically —as  predicted by that kind of an initial bit of  
hand wavy theory I gave you— that you'd expect the  more trees, the lower the error, because the more  
things you're averaging and that's exactly what we  find: the accuracy improves as we have more trees.
John what's up… John: Oh, Víctor… it is possible you  might have just answered his question  
actually as he typed it— but he's asking  on the same theme: the number of trees in   a Random Forest. Does increasing the number  of trees always translate to a better error?
Jeremy: yes it does, always, I mean: tiny bumps,  right? but yeah, once you smoothed it out.  
But decreasing returns and… if you end  up productionizing a Random Forest, then,  
of course, every one of these trees, you have  to, you know, go through for, at inference time,  
so it's not that there's no cost, I mean,  having said that, zipping through a binary  
tree is the kind of thing you can really do fast,  in fact, it's it's quite easy to like literally  
spit out C++ code with a bunch of if statements  and compile it and get extremely fast performance.  
I don't often use more than 100  trees, this is a rule of thumb…  
is that the only one John? okay
What is OOB
So, then there's another interesting  feature of Random Forests which is:   remember how in our example we trained  with 75% of the data on each tree,  
so that means for each tree there was 25% of the  data we didn't train on. Now this actually means  
if you don't have much data, in some situations  you can get away with not having a validation set,  
and the reason why is because for each tree we  can pick the 25% of rows that weren't in that tree  
and see how accurate that tree was on those rows  and we can average for each row their accuracy on  
all of the trees in which they were not part of  the training, and that is called the Out-of-Bag  
Error or OOB error. And this is built in, also,  scikit-learn you can ask for an OOB prediction…
Uhm, John! John: Just before we move on,  Zakia has a question about bagging:  
so we know that bagging is powerful as an  ensemble approach to machine learning, would   it be advisable to try out bagging being first  when approaching a particular, say, tabular task,  
before deep learning? So that's  the first part of the question,   and the second part is: could we create a bagging  model which includes fast.ai deep learning models?
Jeremy: Yes, absolutely. So, to be  clear, you know, bagging is kind of like  
a meta method, it's not a prediction, it's not a  method of modeling itself, it's just a method of  
combining other models. So Random Forests, in  particular, as a particular approach to bagging.  
is a, you know, I would probably always start,  personally, a tabular project with a Random  
Forest because they're nearly impossible  to mess up, and they give good insight,   and they give a good base case. But  yeah, your question then about “can  
you bag other models?” is a very interesting  one, and the answer is: you absolutely can.  
And people very rarely do, but we will,  we will, quite soon, maybe even today.
Model interpretation
So I, you know… you might be getting the  impression I'm a bit of a fan of Random Forests,   and (~before I was…) before, you know, people  thought of me as the Deep Learning guy, people  
thought of me as the Random Forest guy. I used to  go on about Random Forests all the time and one of  
the reasons I'm so enthused about them isn't just  that they're very accurate or (~that they require,   you know…) that they're very hard to mess up and  require very little (~processing…) pre-processing,  
but they give you a lot of quick and easy insight.  And specifically these are the five things  
which I think that we're interested  in and all of which are things that   random for us are good at. They will tell  us how confident are we in our predictions  
on some particular row? So when somebody, you  know… when we're giving a loan to somebody  
we don't necessarily just want to know “How  likely are they to repay?” but we'd also like  
to know “How confident are we that we know?”  because if we're… if we're like well we think  
they'll repay but we're not confident of that we  would probably want to give them less of a loan.  
And another thing that's very important is  when we're then making a prediction… so again,   for example, for credit… let's say you rejected  that person's loan… “Why?...” And a Random Forest  
will tell us “What what is the… what is the reason  that we made a prediction.” And you'll see why,  
and all these things. Which columns are the  strongest predictors - you've already seen   that one, right, that's the Feature Importance  Plot. Which columns are effectively redundant  
with each other, i.e. they're basically  highly correlated with each other.
And then one of the most important ones…  As you vary a column how does it vary   the predictions? So for example in your  credit model, how does your prediction of  
risk vary as you vary… (well something that  probably the regulator would want to know might  
be some, you know…) some protected variable  like you know race or some socio-demographic  
characteristics that you're not allowed to use in  your model. So they might check things like that.
For the first thing: “How confident are we  in our predictions using a particular row   of data?” There's a really simple thing we can  do which is… remember how when we calculated  
our predictions manually we stacked up the  predictions together and took their mean?   Well what if you took their  standard deviation instead?  
So if you stack up your predictions and take  their standard deviation, and if that standard  
deviation is high, that means all of them (all of  the trees) are predicting something different!!  
And that suggests that we don't really know what  we're doing. And so that would happen if different   subsets of the data end up getting completely  different trees for this particular row. So  
there is (like…) a really simple thing you can  do to get a sense of your prediction confidence.
Okay, “Feature Importance” we've already discussed
After I do feature importance (you know…)  like I said when I had the (what) 7,000 or   so columns I got rid of like all-but  30. That doesn't tend to improve the  
predictions of your Random Forest very much,  if at all, but it certainly helps (like,  
you know…) kind of logistically thinking about  cleaning up the data, you can focus on cleaning   those 30 columns, and stuff like that. So I  tend to remove the low importance variables.
Removing the redundant features
I'm going to skip over this bit about removing  redundant features because it's a little bit   outside what we're talking about, but  definitely check it out in the book,  
something called a “Dendrogram.”  But what I do want to mention   is the “Partial Dependence.”  This is the thing which says,  
What does Partial dependence do
“What is the relationship between a column and  the dependent variable - and so this is something  
called a “Partial Dependence Plot.” Now this  one's actually not specific to Random Forests.   A partial dependence plot is something you can  do for basically any machine learning model.  
Let's first of all look at one and then talk  about how we make it. So in this dataset we're  
looking at the relationship… we're looking at  the sale price at auction of heavy industrial  
equipment like bulldozers - this is specifically  the blue books for bulldozers Kaggle competition.  
And a partial dependence plot between the  year that the bulldozer -or whatever was made-  
and the price it was sold for  (this is actually the log price)   is that it goes up. More recent bulldozers… more  recently made bulldozers are more expensive.  
And as you go back… back to older and older  bulldozers, they're less and less expensive,  
to a point. And maybe these ones are some  classic bulldozers you pay a bit extra for.
Now you might think that you could easily create  this plot by simply looking at your data at each  
year and taking the average sale price, but  that doesn't really work very well. I mean  
it kind of does, but it kind of doesn't. Let me  give an example. It turns out that one of the  
biggest predictors of sale price for industrial  equipment is whether it has air conditioning  
and so air conditioning is, you know… it's  an expensive thing to add and it makes the  
equipment more expensive to buy. And most things  didn't have air conditioning back in the 60s and  
70s and most of them do now. So if you plot  the relationship between year made and price  
you're actually going to be seeing a whole  bunch of when… you know… “How popular was  
air conditioning?” right, so you get  this this cross-correlation going on. But we just want to know… what's just the impact  of the year it was made all else being equal.  
So there's actually a really easy way to  do that, which is: We take our data set  
~(we take the we…) and we leave it exactly as  it is, so just use the training dataset… but   we take every single row - and for the year made  column we set it to 1950. And so then we predict  
for every row what would the sale price of that  have been if it was made in 1950, and then we  
repeat it for 1951, and then repeat it for 1952  and so forth, and then we plot the averages.  
And that does exactly what I just said, remember  I said the special words, “all else being equal…”  
This is setting everything else equal. It's  the… Everything else is the data as it actually   occurred and we're only varying YearMade.  And that's what a partial dependence plot is!  
That works just as well for Deep Learning or  Gradient Boosting Trees or Logistic Regressions  
or whatever. It's a really cool thing you  can do. And you can do more than one column  
at a time, you know… You can do two-way  partial dependence plots, for example.
Can you explain why a particular prediction is made
Okay so then another one I mentioned was: “Can  you describe why a particular prediction was  
made? So how did you decide for this particular  row to predict this particular value?” And  
this is actually pretty easy to do - there's  a thing called Tree Interpreter but we could   you could easily create this in about  a half a dozen lines of code. All we do  
is we're saying, okay this customers  come in, they've asked her a loan,  
we put in all of that data through the  Random Forest, spat out a prediction…  
We can actually have a look and say okay, “Well  that in Tree #1 what's the path that went down  
through the tree to get to the leaf node?” And we  can say, oh well, first of all it looked at sex,   and then it looked at postcode, and then  it looked at income, and so we can see  
exactly in Tree #1 which variables were used and  what was the change in ginni for each one. And  
then we could do the same in Tree #2, same in Tree  #3, same in Tree #4… Does this sound familiar?   It's basically the same as our  Feature Importance Plot, right,  
but it's just for this one row of data. And  so that will tell you basically the Feature   Importances for that one particular prediction.  And so then we can plot them, like this.
So for example, this is an example  of an auction price prediction,  
and according to this plot, you know,  so we predicted that the net would be…  
(oh, this is just a change from…) So I  don't actually know what the price is,   but this is how much each one  impacted the price. So Year Made,  
I guess this must have been an older tractor -  it caused our prediction of the price to go down.  
But then it must have been a larger machine  - the Product Size caused it to go up,   a Coupler System made it go up, Model ID made it  go up, and so forth, right. So you can see the  
Reds says this made our prediction go down. Green  made our prediction go up. And so overall you can  
see which things had the biggest impact on the  prediction and what was the direction for each  
one. So it's basically a feature importance plot  but just ~(for a single roll) for a single row
Any questions John? John: Yeah, there are a couple that have, that  have sort of queued up, this is a, this is a  
good spot to, to jump to them. So, first of all,  Andrew is asking, jumping back to the OOB era:  
“would you ever exclude a tree from a forest  you've had if it had a bad Out of Bag Error?”  
Like if you, if you have a bogus, if you  had a particularly bad tree in your ensemble Jeremy: Yeah…
John: Might you just drop… Would you delete a tree   that was not doing its thing?  It's not playing its part.
Jeremy: No you wouldn't. If you start  deleting trees then you are no longer  
having an unbiased prediction of the  dependent variable. You are biasing it  
by making a choice so, even the bad ones, will  be improving the quality of the overall average.
John: All right, thank you. Zakia followed up with  the question about bagging and we're just going,  
you know, layers and layers here, you know: we  could go on and create ensembles of bagged models?  
and, you know, is it reasonable to  assume that they would continue… Jeremy: So that's not going to make much  difference, right? If they're all like…  
you could take your 100 trees, split them into  groups of ten, create ten bagged ensembles and  
then average those but the average of an average  is the same as the average. You could like have  
a wider range of other kinds of models, you could  have like Neural Nets trained on different subsets   as well but again, it's just the average of  an average, we'll still give you the average.
John: Right, so there's not a lot of value in,  kind of, structuring the ensemble you just…
Jeremy: I mean, some, some  ensembles you can structure,   but not bagging. Bagging's the simplest  one, it's the one I mainly use,  
there are more sophisticated approaches  but this one is nice and easy. John: All right, and there's one that is a bit  specific and it's referencing content you haven't  
covered but we're here now so... and it's on  explainability so, feature importance of Random  
Forest models sometimes has different results when  you compare it to other explainability techniques  
like SHAP, S-H-A-P, or LIME. And  we haven't covered these in the  
course but Amir is just curious if  you've got any thoughts on which is   more accurate or reliable: Random Forest  feature importance or other techniques
Jeremy: I would lean towards, more immediately,  trusting Random Forest Feature Importances over  
other techniques on the whole, on the basis  that it's very hard to mess up a Random Forest.  
So, yeah, I feel like pretty confident that  a Random Forest Feature Importance is gonna  
be pretty reasonable as long as this is the kind  of data which a Random Forest is likely to be  
pretty good at, you know, doing, you know, if it's  like a computer vision model Random Forest aren’t  
particularly good at that and so, one  of the things that Breiman talked about   a lot was explainability and he's got  a great essay called the two cultures  
of statistics (“Statistical Modeling: The  Two Cultures”) in which he talks about —I   guess what nowadays call kind of like data  scientists, machine learning folks versus  
classic statisticians—. And he was, you know,  definitely a data scientist well before the  
label existed. And he pointed out, yeah,  you know, first and foremost you need a  
model that's accurate. It needs to make good  predictions. A model that makes bad predictions,  
will also be bad for making explanations because  it doesn't actually know what's going on.  
So if, you know, if you, if you've got a Deep  Learning model that's far more accurate than your   Random Forest then it's, you know, explainability  methods from the Deep Learning model world  
probably be more useful because it's  explaining a model that's actually correct.
All right, let's take a 10 minute break  and we'll come back at five past seven.
Can you overfit a random forest
Welcome back, one person pointed  out I noticed I got the chapter   wrong — it's Chapter 9 not Chapter  8 in the book, I guess I can't read.  
Somebody asked during the break  about overfitting. Can you overfit  
a Random Forest?. Basically no, not really,  adding more trees will make it more accurate.  
It kind of asymptotes so you can't make it  infinitely accurate by using infinite trees but  
certainly you know adding more trees won't make it  worse. If you don't have enough trees and you let  
the trees grow very deep: that could overfit. So  you just have to make sure you have enough trees.
Radek told me about this experiment he did  during… Radek talked to me during the break   about an experiment he did, which is something  I've done something similar, which is adding  
lots and lots of randomly generated columns to  a data set and try to break the Random Forest.  
And if you tried it, it basically doesn't work,  it's like, it's really hard to confuse a Random  
Forest by giving it lots of meaningless data,  it does an amazingly good job of picking out  
the useful stuff. As I said, you know, I had 30  useful columns out of 7,000 and it found them  
perfectly well. And often, you know, when you  find those 30 columns, you know, you could go to,  
you know —I was doing consulting at the time—  go back to the client and say like: tell me   more about these columns, and say like: “oh! well  that one there we've actually got a better version  
of that now, there's a new system, you know, we  should grab that… and, oh, this column actually   that was because of this thing that happened  last year but we don't do it anymore…” or,  
you know, like you can really have this kind of  discussion about the stuff you've zoomed into.
You know, there are other things that you  have to think about with lots of kinds of   models like, particularly regression  models; things like interactions.  
You don't have to worry about that  with Random Forests like, because you   split on one column and then split on another  column you get interactions for free as well.
Normalization you don't have to worry about, you  know you don't have to have normally distributed   columns. So yeah, definitely worth a try.
Now something I haven't gone  into… is Gradient Boosting.  
What is gradient boosting
But if you go to explain.ai, you'll  see that my friend Terence and I have  
a three-part series about Gradient Boosting,  including pictures of golf made by Terence.  
But to explain Gradient Boosting is a  lot like Random Forests but rather than  
fitting a tree again and again and again  on different random subsets of the data…  
instead, what we do is we fit very, very,  very small trees, so hardly ever any splits  
and we then say: “okay, what's the error”. So,  you know, so, imagine the simplest tree would be  
our “OneR” raw tree of male versus female, say,  and then you take what's called the residual:  
that's the difference between the prediction and  the actual, it's the error. And then you create   another tree, which attempts to predict that —a  very small tree— and then you create another very  
small tree which tries to predict the error from  that and so forth, right? Each one is predicting  
the residual from all of the previous ones. And so  then to calculate a prediction, rather than taking  
the average of all the trees, you take the sum  of all the trees, because each one is predicting  
the difference between the actual and all of  the previous trees. And that's called boosting  
—versus bagging— so boosting and bagging  are two kind of meta ensembling techniques,  
and when bagging is applied to trees it's  called a Random Forest and when boosting is  
applied to trees it's called a Gradient Boosting  Machine or Gradient Boosted Decision Trees.
Gradient Boosting is, generally speaking,  more accurate than Random Forests,  
but you can absolutely overfit and so  therefore it's not necessarily my first  
go-to thing. Having said that there are ways to  avoid overfitting, but yeah, (it's just, it's not,  
you know) because it's breakable, it's not my  first choice. But yeah, check out our stuff  
here if you're interested and, you know, there  is stuff which largely automates the process,  
there's lots of hyper parameters you have  to select. People generally just, you know,   try every combination of hyper parameters and,  in the end, you generally should be able to get  
a more accurate Gradient Boosting model than  Random Forest… but not necessarily by much.
Okay, so that was the  
Introducing walkthrus
Kaggle notebook on Random Forests:  “How random forests really work”.
So, what we've been doing is having  this daily walk-through where me and  
—I don't know how many— 20 or 30 folks get  together on a Zoom call and chat about,  
you know, getting through the course, and  setting up machines, and stuff like that. And,  
you know, we've been trying to kind of  practice what, you know, things along the way   and so, a couple of weeks ago, I wanted to show:  
what does it look like to pick a Kaggle  competition and just like do the normal, sensible,  
kind of mechanical steps that you  would do for any computer vision model.
And so the competition I picked  was Paddy Disease Classification,  
which is about recognizing diseases,  rice diseases and rice paddies.  
And yeah I spent —I don't know— a couple of  hours, or three —I can't remember—. A few hours,  
throwing together something and I found that I  was number one on the leaderboard and I thought:  
“oh, that's interesting”, like, because you never  quite have a sense of how well these things work.  
And then I thought: “well there's all these other  things we should be doing as well”, and I tried   three more things, and each time I tried another  thing, I got further ahead at the top of the  
leaderboard. So I thought it'd be cool to take you  through the process. I'm going to do it reasonably  
quickly because the walkthroughs are all available  for you to see the entire thing in, you know,  
seven hours of detail —or however long we probably  were, six to seven hours of conversations—  
but I want to kind of take you through  the basic process that I went through.
What does fastkaggle do
So since I've been studied to do more stuff  on Kaggle, you know, I realized there's some  
kind of manual steps I have to do  each time, particularly because I   like to run stuff on my own machine  and then kind of upload it to Kaggle.  
So to do, to to make my life easier I created  a little module called fastKaggle which you'll  
see in my notebooks from now on, which  you can download from pip or conda,  
and as you'll see it makes some things a bit  easier. For example: downloading the data for   the Paddy Disease Classification, if you just  run “setup_comp()” and pass in the name of  
the competition, if you are on Kaggle, it will  return a path to that competition data that's  
already on Kaggle; if you are not on Kaggle and  you haven't downloaded it, it will download and   unzip the data for you. If you're not on Kaggle  and you have downloaded and unzip the data,  
it will return a path to the one that you've  already downloaded. Also if you are on Kaggle   you can ask it to make sure that pip things are  installed —that might not be up to date otherwise—  
so that's basically one line of code  now gets us all set up and ready to go.   So this path… so I ran this particular one on  my own machine, so it downloaded and unzipped  
the data. I've also got links to the six  walkthrus so far: these are the videos. Oh yes,  
and here's my result after these four attempts,  that's a few fiddling around at the start.  
So the overall approach at, is, well and  this is not just a Kaggle competition,  
right? the reason I like looking  at Kaggle competitions is   you can't hide from the truth in a Kaggle  competition, you know, when you're working on  
some work project or something, you might be able  to convince yourself and everybody around you that  
you've done a fantastic job of not overfitting,  and that your model's better than what anybody  
else could have made, and whatever else; but  the brutal assessment of the private leaderboard  
will tell you the truth. Is your model actually  predicting things correctly? and, is it overfit?
Until you've been through that process,  you know, you're never going to know. And   a lot of people don't go through that process  because at some level they don't want to know.  
But it's okay, you know, nobody needs it,  you don't have to put your own name there.  
I always did, right from the very first one  I wanted, you know, if I was going to screw   up royally I wanted to have the pressure on  myself of people seeing me in last place.  
But you know, it's fine, you can do it  all anonymously and you'll actually find  
as you improve you'll have so much  self-confidence, you know. And the  
stuff we do in a Kaggle competition is indeed a  subset of the things we need to do in real life,  
but it's an important subset, you know. Building  a model that actually predicts things correctly  
and doesn't overfit is important. And furthermore  structuring your code and analysis in such a way  
that you can keep improving over a three-month  period without gradually getting into more and   more of a tangled mess of impossible to understand  code and having no idea what UntitledCopy13 was,  
and why it was better than 25, right, this is  all stuff you want to be practicing, ideally  
well away from customers or whatever, you  know, before you've kind of figured things out.
So the things I talk about here about doing  things well in this Kaggle competition  
should work, you know, in other settings as well.  And so these are the two focuses that I recommend:  
Get a really good validation set together - we've  talked about that before, right, and in a Kaggle  
competition (that's like…) it's very rare to see  people do well in a Kaggle competition who don't   have a good validation set. Sometimes that's  easy, and this competition actually it is easy,  
because the the the test set seems to be a random  sample, but most of the time it's not actually,  
I would say. And then how quickly can you iterate?  How quickly can you try things and find out what  
worked? So obviously you need a good validation  set otherwise it's impossible to iterate.  
And so “quickly iterating” means  not saying what is the biggest,  
you know, OpenAI takes four months  on 100 TPUs model that I can train.  
It's what can I do that's going to train in a  minute or so and will quickly give me a sense  
of… like well I could try this, I could try  that, what things going to work, and then try,  
you know, 80 things. It also doesn't mean that  saying like… Oh I heard this is amazing new  
Bayesian hyper parameter tuning approach, I'm  going to spend three months implementing that,   because that's gonna, like, give you one thing.  But actually to do well in these competitions or  
in machine learning in general, you actually have  to do everything reasonably well. And doing just  
one thing really well will still put you somewhere  about last place. So I actually saw that a couple  
of years ago, an Aussie guy who's a very very  distinguished machine learning practitioner,  
actually put together a team and entered a Kaggle  competition and literally came in last place,  
because they spent the entire three months  trying to build this amazing new fancy thing and  
never actually, never actually iterated. If you  iterate I guarantee you won't be in last place.
Okay, so here's how we can grab our data,   with FastKaggle, and it gives  us (tells us) what path it's in.  
And then I set my random seed - and I only do  this because I'm creating a notebook to share.  
You know, when I share a notebook I like to be  able to say “as you can see this is 0.83 blah blah   blah right and know that when you see it, it'll be  0.83 as well. But when I'm doing stuff otherwise,  
I would never set a random seed. I want to be able  to run things multiple times and see how much it   changes each time, all right, because that'll  give me a sense of like… are the modifications  
I'm making, changing it because they're improving  it or making it worse, or is it just random   variation. So if you (or if you) always set  a random seed, that's a bad idea because you  
won't be able to see the random variation. So  this is just here for presenting a notebook.
Okay, so the data they've given us, as usual,  they've got a sample submission, they've got  
some test set images, they've got some training  set images, a CSV file about the training set  
and then these other two you can  ignore because I created them.   So let's grab a path to train  images… and so do you remember  
get_image_files()... so that gets us a list of  the file names of all the images here recursively,  
so we could just grab the first  one, and take a look - so it's 480   by 640. Now we've got to be careful. This is a  pillow image (Python Imaging Library image.) In  
the imaging world they generally say columns  by rows. In the array slash tensor world we  
always say rows by columns. So if you ask pytorch  what the size of this is it'll say 640 by 480,  
and I guarantee at some point this is going  to bite you. So try to recognize it now.
Okay so they're kind of taller than they are…  at least this one is taller than it is wide.  
fastcore.parallel
So I actually like to know  were they all this size,   because it's really helpful if they all  are all the same size, or at least similar.  
Believe it or not the amount of time it takes  to decode a JPEG is actually quite significant  
and so figuring out what size these things  are is actually going to be pretty slow,  
but my fastcore library has a parallel sub module  which can basically do anything that you can do  
in Python, it can do it in parallel. So in this  case we wanted to create a pillow image and get   its size. So if we create a function that does  that and pass it to parallel, passing in the  
function and the list of files, it does it in  parallel and that actually runs pretty fast.  
And so here is the answer… (I don't know how  this happened…) 10,403 images are indeed 480 by  
640 and four of them aren't. So basically what  this says to me is that we should pre-process  
them or you know at some point process them  so that they're probably all 480 by 640,   or all basically kind of the same size.  We'll pretend they're all this size  
but we can't not do some initial resizing  otherwise this is going to screw things up.
item_tfms=Resize(480, method='squish')
So like the probably the easiest way to do  things, the most common way to do things,   is to either squish or crop  every image to be a square.  
So squishing is when you just… (in this  case…) squish the aspect ratio down as  
opposed to cropping randomly a section out. So if  we call Resize() ‘squish’ it will squish it down,  
and so this is 480 by 480 squared. So this is  what it's going to do to all of the images first,  
on the CPU, that allows them to be all  batched together into a single mini batch  
(everything in a mini batch has to be the  same shape otherwise the GPU won't like it,)  
and then that many batches put through data  augmentation, and it will grab a random subset  
of the image, and make it a 128 by 128 pixels.  And here's what that looks like, here's our data.  
So show_batch() works for pretty much  everything, not just in the fast AI library,   but even for things like FastAudio which are  kind of community based things. You should  
better use show_batch() on anything and see, or  hear, or whatever, what your data looks like.
I don't know anything about  rice disease but apparently   these are various rice diseases  and this is what they look like.
So I jump into creating models much more quickly  than most people, because I find… model, you know,  
models… are a great way to understand my data,  as we've seen before. So I basically build a  
model as soon as I can, and I want to create a  model that's going to let me iterate quickly.  
So that means that I'm going to need a model that  can train quickly. So Thomas Capell and I recently  
Fine-tuning project
did this big project “The Best Vision Models For  Fine Tuning” where we looked at nearly a hundred  
different architectures from Ross Whiteman's  Timm Library (Pytorch Image Model Library)  
and looked at which ones could we fine-tune -  which ones had the best transfer learning results.  
And we tried two different datasets, very  different datasets. One is the pets dataset   that we've seen before - so trying to predict  what breed of pet is from 37 different breeds.  
And the other was a satellite  imagery data set called planet.  
So very very different data sets in terms of  what they contain and also very different sizes.  
The planet one is a lot smaller the Pets  one is a lot bigger. And so the main things  
Criteria for evaluating models
we measured were how much memory did it use, how  accurate was it, and how long did it take to fit.  
And then I created this score which can… which  combines the fit, time and error rate together.
And so this is a really useful table for picking  a model. And now in this case I want to pick  
something that's really fast, and there's one  clear winner on speed, which is resnet26d.  
And so its accuracy was 6% versus the best was  like 4.1% - so okay it's not amazingly accurate,  
but it's still pretty good, and it's going to be  really fast, so that's why I picked “resnet26d.  
A lot of people think that when they do deep  learning they're going to spend all of their time  
learning about exactly how a resnet26d is made and  convolutions and resonant blocks and transformers  
and blah blah blah. We will cover all that stuff  in Part Two and a little bit of it next week  
but it almost never matters, right, it's  just a function, right, and what matters  
is the inputs to it, and the outputs to it,  and how fast it is, and how accurate it is.
So let's create a learner which with  a resnet26d from our data loaders,  
and let's run lr_find(). So lr_find() will put  through one mini-batch at a time starting at a  
very very very low learning rate, and gradually  increase the learning rate, and track the loss.  
And initially the loss won't improve because the  learning rate is so small it doesn't really do  
anything, and at some point the learning rate is  high enough that the loss will start coming down,   and then at some other point the learning rate  is so high that it's going to start jumping past  
the answer and it's going to predict worse.  And so somewhere around here is a learning  
rate we'd want to pick. We've got a couple  of different ways of making suggestions.  
I generally ignore them because these suggestions  are specifically designed to be conservative.  
They're a bit lower than perhaps optimal, in  order to make sure we don't recommend something   that totally screws up, but I kind of like  to say, like well… How far right can I go and  
still see it, like, clearly really improving  quickly, and so I'd pick somewhere around  
0.01 for this. So I can now fine-tune  our model with a learning rate of 0.01,  
three epochs, so look!... the whole thing took  a minute! That's what we want, right, we want to   be able to iterate, rapidly, just a minute or so.  So that's enough time for me to go and you know,  
grab a glass of water, or do some reading,  like I’m not going to get too distracted.  
And what did we do before we submit?  Nothing! We submit as soon as we can. Okay,  
Should we submit as soon as we can
let's get our submission in. So we've got  a model, let's get it in. So we read in  
our CSV file of the sample submission and  so the CSV file basically looks like we're  
going to have to have a list of the image file  names, in order, and then a column of labels.
So we can grab all the image files in the  test image —like so— and we can sort them.  
And so now we want is —what  we want is— a dataloader,   which is exactly like the dataloader we use to  train the model, except pointing at the test set,  
we want to use exactly the same transformations  so there's actually a dls.test_dl() method  
—which does that— you just pass in the  new set of items, so the test set files.  
So this is a dataloader which we can use for our  test set. A test dataloader has a key difference  
to a normal data loader which is that it does  not have any labels. So that's a key distinction.
So we can get the predictions  for our learner, passing in   that dataloader and in the case of a  classification problem you can also  
ask for them to be decoded, decoded means rather  than just get returned the probability of every  
rice disease for every class, it'll tell you what  is the index of the most probable rice disease,  
that's what decoded means. So this is returning  the probabilities, targets —which obviously will  
be empty because it's a test set, so throw  them away— and those decoded indexes which   look like this: numbers from naught (0) to nine  (9) because there's ten possible rice diseases.  
The Kaggle submission does not expect numbers  naught (0) to nine (9), it expects to see  
strings like these. So, what do those numbers  from naught (0) to nine (9) represent?  
We can look up our vocab to get a list, so  that's zero, that's one, et cetera, that's nine.  
So, I realized later this is a  slightly inefficient way to do it,   but it does the job I need to  be able to map these to strings  
so, if I enumerate the vocab, that gives me pairs  of numbers; zero (0): bacterial leaf blight,  
one (1): bacterial leaf streak, etc. I  can then create a dictionary out of that,  
and then I can use pandas to look up each thing  in a dictionary: they call that map. If you're a  
pandas user you've probably seen map used before  being passed a function —which is really, really  
slow— but if you pass map a dict it's actually  really, really fast, so do it this way if you can.
So here's our predictions, so we've got our  
submission sample: submission file “ss”, so if  we replace this column label with our predictions  
—like so— then we can turn that into a CSV,  
and remember, this means, this means:  run a bash command —a shell command—,  
head is the first few rows —let's just  take a look. That looks reasonable!
So we can now submit that to Kaggle. Now,  iterating rapidly means everything needs to be  
fast and easy. Things that are slow and hard  don't just take up your time but they take up your  
mental energy, so even submitting to Kaggle needs  to be fast. So I put it into a cell, so I can just  
run this cell: api.competition_submit_cli(), this  CSV file, give it a description, so just run the  
cell and it submits to Kaggle. And as you can see  it says: here we go!, “successfully submitted”.  
So that submission was terrible: top eighty  percent also known as bottom twenty percent,  
which is not too surprising, right?, I  mean, it's one minute of training time.
But it's something that we can  start with and that would be like:   however long it takes to get to this point,  that you put in our submission, now you've  
really started, right? because then tomorrow  you can try to make a slightly better one.
How to automate the process of sharing kaggle notebooks
So I'd like to share my notebooks  and so, even sharing the notebook,   I've automated. So part of fastkaggle is:  you can use this thing called push_notebook  
and that sends it off to Kaggle to create…  
a notebook on Kaggle, there  it is, and there's my score.
As you can see it's exactly the same thing.
Why would you create public notebooks on Kaggle?  Well, it's the same brutality of feedback that  
you get for entering a competition, but this time  rather than finding out, in no uncertain terms,  
whether you can predict things accurately; this  time you can find out —no it's no uncertain   terms— whether you can communicate things in  the way that people find interesting and useful.  
And if you get zero votes, you know, so be  it, right? that's something to know and then,  
you know, ideally go and ask some friends,  like: what do you think I could do to  
improve? and if they say: oh, nothing, it's  fantastic! you can tell: no, that's not true,   I didn't get any votes, I'll try again, this  isn't good, how do I make it better, you know,  
and you can try and improve because if you  can create models that predict things well,  
and you can communicate your results  in a way that is clear and compelling,   you're a pretty good data scientist, you  know, like they're two pretty important  
things and so here's a great way to test  yourself out on those things and improve.
Yes John. John: yes Jeremy we have a sort of  —I think— a timely question here  
from Zakia about your iterative  approach. And they're asking:  
“do you create different Kaggle  notebooks for each model that you try?” Jeremy: yeah… John: so one Kaggle book for the first one,  then separate notebooks subsequently, or do you,  
do append to the bottom of us — Jeremy:  yeah, yeah — but what's your strategy?  Jeremy: that's a great question.  And I know Zaki is going through the  
daily walkthroughs but isn't quite caught  up yet so, I would say: keep it up because   in the six hours of going through this  you'll see me create all the notebooks…
but, if I go to the actual directory I used,  
you can see them. So basically: yeah. I started  with, you know, what you just saw, a bit messier  
without the pros, but that same basic thing, I  then duplicated it to create the next one —which  
is here— and because I duplicated it, you know,  this stuff which I still need it's still there,   right? and so I run it. And I don't always know  what I'm doing, you know, and so, at first,  
if I don't really know what I'm doing next  I'm going to duplicate it, it will be called,   you know, “first steps in the road to the top part  one dash copy one”, you know, and that's okay.  
As soon as I can I'll try to rename that,  once I know what I'm doing, you know,  
or if it doesn't seem to go anywhere I'll rename  it into something like, you know, experiment  
blah blah blah and I'll put some notes at the  bottom and I might put it into a failed folder   or something. But yeah, it's like, it's a very low  tech approach that I find works really well: which  
is just duplicating notebooks and editing them and  naming them carefully and putting them in order  
and, you know, put the file name in when you  submit as well. And then of course also if  
you've got things in git, you know, you can have  a link to the git commit so you'll know exactly   what it is… generally speaking from me, you know,  my notebooks will only have one submission in and  
then I'll move on and create a new notebook so  I don't really worry about versioning so much,  
but you can do that as well if that helps you. Yeah, so that's basically what I do  and and I've worked with a lot of  
people who use much more sophisticated and  complex processes and tools and stuff but  
none of them seem to be able to stay as well  organized as I am, I think they kind of get   a bit lost in their tools, sometimes. And  file systems and file names I think are good.
John: oh great thanks so, away from that kind  of dev process, more towards the specifics of,  
you know, finding the best model and all that  sort of stuff. We've got a couple of questions   that are in the same space, which is, you know,  we've got some people here talking about AutoML  
AutoML
Frameworks which you might want to, you know,  touch on for people who haven't heard of those.   If you've got any particular AutoML Frameworks  you think are worth recommending. Or just,  
more generally, how do you go trying  different models. Random Forest,   Gradient Boosting, Neural Networks… it  just so in that space if you can comment it
Jeremy: sure. I use AutoML less than anybody  I know, I would guess, which is to say: never.  
Hyper parameter optimization: never. And the  reason why is I like being highly intentional,  
you know, I like to think more like a scientist  and have hypotheses and test them carefully  
and come up with conclusions —which then  I implement, you know—. So for example,  
in this “best vision models for fine tuning” I  didn't try a huge “grid search” of every possible  
model, every possible learning rate, every  possible pre-processing approach blah blah   blah right. Instead, step one was to find  out: well, which things matter, right? So,  
for example, “does whether we squish or crop make  a difference?” you know, “are some models better  
with squish and some models better with crop?”  and so we just chested that for… and again,  
not for every possible architecture but for one  or two versions of each of the main families,   that took: 20 minutes. And the  answer was: no, in every single case  
the same thing was better. So we don't  need to do a grid search over that anymore,   you know. Or another classic one is like, learning  rates, most people do a, kind of, grid search  
over learning rates or they'll train a thousand  models, you know, with different learning rates.   But this fantastic researcher named Leslie Smith  invented the learning rate finder a few years ago,  
we implemented it, I think, within days of  it (first) coming out as a technical report,  
and that's what I've used ever since: it  works well and runs in a minute or so.
Yeah, I mean, then like Neural Nets versus  GBMs versus Random Forests, I mean, that's…  
that shouldn't be too much of a question  on the whole, like: they have pretty clear  
places that they go. Like, if I'm doing  computer vision I'm obviously going to use a  
computer vision Deep Learning model and which one  I would use, well, if I'm transfer learning —which  
hopefully is, always— I would look up the two  tables here: this is my table for pets which is,  
which are the best at fine tuning to very  similar things to what they're pre-trained on,  
and then the same thing for planet is: “which ones  are best for fine tuning for two data sets that  
are very different on what they're trained on”?  And it happens in both case they're very similar,   in particular “convnext” is right  up towards the top in both cases,  
so I just like to have these rules of thumb  and… yeah, my rule of thumb for tabular is:  
Random Forest is going to be the fastest  easiest way to get a pretty good result, GBMs  
probably going to give me a slightly better result  if I need it, and can be bothered fussing around
GBM I would probably, yeah, actually  I probably would run a hyper parameter  
sweep because it is fiddly and  it's fast, so you may as well.
Why the first model run so slow on Kaggle GPUs
So, yeah, you know, we were able to  make a slightly better submission,  
a slightly better model. And so, I had a couple  of thoughts about this. The first thing was:  
that thing trained in a minute  on my home computer and then,  
when I uploaded it to Kaggle, it took about  four minutes per epoch —which was horrifying—.  
And, Kaggle GPUs are not amazing but they're  not that bad so I knew something was up,  
and what was up is I realized that they only  have two virtual CPUs which nowadays is tiny,  
like, you know, you generally want —as a rule  of thumb— about eight physical CPUs per GPU.
And so, spending all of its time just reading the  damn data. Now, the data was 640 by 480 and we  
were ending up with only 128 pixel size bits for  speed. So there's no point doing that every epoch  
so, step one was to make my  Kaggle iteration faster, as well,  
and so a very simple thing to do: resize  the images, so fastai has a function called  
resize_images() and you say: “okay, take all the  train images and stick them in the destination,  
making them this size recursively”, and that will  recreate the same folder structure over here.  
And so that's why I call this the training  path: because this is now my training data  
and so, when I then, trained on that, on Kaggle,  it went down to four times faster with no loss of  
accuracy so, that was kind of step one was:  to actually get my first iteration working.  
Now, still, I bet it's a long time and on Kaggle  you can actually see this little graph showing:  
how much the CPU is being used, how much  the GPU is being used; on your own home   machine you can —there are tools free GPU—  you know, free tools to do the same thing.  
I saw that the GPU was still hardly being used  so it's still CPU was being driven pretty hard,  
I wanted to use a better model anyway —to  move up the leaderboard— so I moved from a…  
Oh, by the way, this graph is very useful, so this  is… this is speed versus error rate by family.  
And so we're about to be  looking at these convnext models  
so we're going to be looking  at this one convnext_tiny…
Here it is convnext_tiny. So we were looking at  resnet26d, which took this long on this data set,  
but this one here is nearly the best (I think  it's third best) but it's still very fast,  
and so it's the best overall score. So let's  use this, particularly because, you know, we're  
still spending all of our time waiting for the CPU  anyway. So it turned out that when I switched my  
architecture to convnext it basically ran just  as fast, on Kaggle, so we can then train that.  
How much better can a new novel architecture improve the accuracy
Let me switch to the Kaggle version because  my outputs are missing for some reason.
So yeah… so I started out by running the resnet26d  on the resized images and got similar error rate,  
but I ran a few more epochs, got 12% error rate.  And so then I do exactly the same thing but with  
convnext_small and 4.5% error rate, so I don't  think that different architectures are just  
tiny little differences - this is over twice as  good. And a lot of folks you talk to will never  
Convnext
have heard of this convnext because it's very  new and I've noticed a lot of people tend not to  
keep up to date with new things. They kind of  learn something at University and then they   stop… stop learning. So if somebody's still  just using resnets all the time, you know,  
you can tell them, we've actually… we've moved on,  you know. Resnets is still probably the fastest  
but for the mix of speed and performance, you  know, not so much. Convnext, you know again,  
you want these rules of thumb, right. If you're  not sure what to do, this convnext, okay,  
and then like most things there's different sizes  - there's a tiny, there's a small, there's a base,  
there's a large, there's an extra large, and you  know it's just… well let's look at the picture.  
This is it here, right, large takes longer  but lower error, tiny takes less time but  
higher error, right, so you pick about  your speed versus accuracy trade-off,  
for you. So for us small is great. And so yeah  now we've got a 4.5% error, that's terrific!
Now let's iterate! On Kaggle this is  taking about a minute per epoch. On  
my computer it's probably taking about  20 seconds per epoch, so not too bad.  
So you know, one thing we could try is  instead of using squish as our pre-processing,  
let's try using crop. So that will randomly  crop out an area, and that's the default,  
so if I remove the “method=squish” that will crop.  So you see how I've tried to get everything into  
a single function, right, the single function. I  can tell it… (let's go and find the definition…)  
what architecture do I want to train, how do  I want to transform the items, how do I want   to transform the batches, and how many epochs  do I want to do - that's basically it, right.  
So this time I want to use the same architecture  convnext, I want to resize without cropping,  
and then use the same data augmentation,  and okay, error rate's about the same.  
So not particularly… it's a tiny bit  worse, but not enough to be interesting.  
How to iterate the model with padding
Instead of cropping, we can pad. Now padding is  interesting… do you see how these are all square,  
right, but they've got black borders…   so padding is interesting because it's the only  way of pre-processing images which doesn't distort  
them and doesn't lose anything. If you crop, you  lose things. If you squish, you distort things.  
This does neither. Now of course the downside  is that there's pixels that are literally  
pointless - they contain zeros. So every way of  getting this working has its compromises but this  
approach of resizing where we pad with zeros is  not used enough - and it can actually often work  
quite well - in this case it was about as good as  our best so far, but no, not huge differences yet.
What does our data augmentation do to images
What else could we do? Well, what we could do is…  see these pictures? This is all the same picture  
but it's gone through our data augmentation,  so sometimes it's a bit darker, sometimes it's  
flipped horizontally, sometimes it's slightly  rotated, sometimes it's slightly warped,   sometimes it's zooming into a slightly different  section, but this is all the same picture.  
Maybe our model would like some of  these versions better than others,   so what we can do is we can  pass all of these to our model,  
get predictions for all of them, and take  the average, right. So it's our own kind of  
like little mini-bagging approach, and  this is called Test Time Augmentation.  
Fast.ai is very unusual in making that  available in a single method. You just  
pass TTA and it will pass multiple augmented  versions of the image and average them for you.
And so this is the same model as before, which had  a 4.5%, so if instead if we get TTA predictions  
and then get the error rate, wait why does it say  4.8?... last time I did this it was way better.  
Well that's messing things up isn't it?  
So when I did this originally on my home  computer it went from like 4.5 to 3.9,  
so possibly I got a very bad luck this  time. So this is the first time I've  
actually ever seen TTA give a worst result.  So that's very weird. I wonder if it's…  
if I should use something other than the  crop-n-padding. All right, I'll have to   check that out, and I'll try and come back to you  and find out why in this case this one was worse.  
How to iterate the model with larger images
Anyway take my word for it every other time I've  tried it TTA has been better, so then you know  
now that we've got a pretty good way of resizing,  we've got TTA, we've got a good training process,  
let's just make bigger images, and something  that's really interesting and a lot of people   don't realize is your images don't have to be  square, they just all have to be the same size,  
and given that nearly all of our images are  640x480 we can just pick, you know, that aspect  
ratio (so for example 256x192) and we'll resize  everything to the same aspect ratio rectangular,  
and that should work even better still.  So if we do that we'll do 12 epochs…  
okay now our error rate is down to 2.2% And then we'll do TTA. Okay this time you can  see it actually improving, down to under 2%  
so that's pretty cool, right… we've got our error  rate… at the start of this notebook we were at  
12% and by the time we've got  through our little experiments  
we're down to under 2%. And nothing  about this is in any way specific to  
rice, or this competition, you  know, it's like this is a very  
mechanistic, you know, standardized approach,  which you can use for certainly any kind of this  
type of computer vision competition - they'd have  computer vision data set almost. But you know, it  
looked very similar for a collaborative filtering  model, a tabular model, NLP model, whatever.
pandas indexing
So, of course, again, I want  to submit as soon as I can. So,   just copy and paste the exact same steps I took  last time basically for creating a submission.
So, as I said last time, we did it using  pandas, but there's actually an easier way.   So the step where here I've got  the numbers from naught to nine,  
which is like, which… which rice disease is it?  so here's a cute idea; we can take our vocab,  
and make it an array so that's going to be a list  of 10 things and then we can index into that vocab  
with our indices which is kind of weird this is a  list of 10 things this is a list of… I don't know  
four or five thousand things? so this will give  me four or five thousand results which is each  
vocab item for that thing? So this is another way  of doing the same mapping, and I would spend time  
playing with this code to understand what it does,  because it's the kind of like very fast, what,  
you know, not just in terms of writing, but this  this… the… this would optimize… you know on on  
the CPU very very well.This is the kind of coding  you want to get used to, this kind of indexing.  
Anyway, so then we can submit it just like  last time, and when I did that I got in the  
top 25 percent, and that's… that's where you  want to be, right? Like generally speaking,  
I find in Kaggle competitions the top 25 percent  is like, you're kind of like solid competent  
level? you know? look it's not to say like, it's  not easy you've got to know what you're doing,  
but if you get in the top 25, and I think you  can really feel like yeah this is… this is a…  
you know very reasonable attempt, and so that's  I think, this is a very reasonable attempt.  
Okay before we wrap up, John any last questions?
John: Um yeah, there's there's two I think that  would be good if we could touch on quickly before  
What data-augmentation does tta use?
you wrap up; one from Victor asking about TTA:  “when I use TTA during my training process do I  
need to do something special during inference or  is this something you use only during validate?” Jeremy: Okay so just to explain TTA means  “test time augmentation” so, specifically it  
means inference. I think you mean augmentation  during training? so yeah… so during training,  
you basically always do augmentation which  means you're varying each image slightly so  
that the model never sees the same image exactly  the same twice, and so it can't memorize it.  
On Fast Ai – and as I say I don't think anybody  else does this as far as I know – if you call TTA,  
it will use the exact same augmentation approach  on whatever data set you pass it, and average out  
the prediction, but… but like multiple times  on the same image, and will average them out,   so you don't have to do anything different, but if  you didn't have any data augmentation in training  
you can't use TTA. It uses the same… by default  the same data augmentation you use for training.
John: Great! thank you! and  the other one is about how  
you know when you first started this example you  squared the models and you the images rather,   and you talked about squashing versus cropping  versus you know, clipping and scaling, and so on,  
but then you went on to say that these models  can actually take rectangular inputs right?  
so there's a question that's kind of probing at  that… you know, if the…” if the models can take  
rectangular inputs why would you ever even  care as long as they're all the same size?”
Jeremy: So, I find most of the time data sets  tend to have a wide variety of input sizes and  
aspect ratios, so you know if there's just as many  tall skinny ones as wide short ones you know you  
doesn't make sense to create a rectangle because  some of them… you're going to really destroy them,   so a square is the kind of best compromise in  some ways there are better things we can do  
which we don't have any off-the-shelf Library  support for yet, and I don't think… I don't know,  
that anybody else has even published about this,  but we've experimented with kind of trying to  
batch things that are similar  aspect ratios together, and use   the kind of median rectangle for those and have  had some good results with that, but honestly  
not 89, 99 percent of people, given a wide variety  of aspect ratios, chuck everything into a square.
John: A follow-up! this is my own  interest, “have you ever looked at  
you know so the issue with with padding as you say  is that you're putting you know black pixels there  
those are not NANs, those are black pixels,  that's right, and so there's something problematic  
to me you know conceptually about that,   you know, when you… when you see for example  four to three aspect ratio footage presented  
for broadcast on 16 to nine you get the kind  of the Blurred stretch? that kind of stuff? Jeremy: No, we played with that a lot yeah? I  used to be really into it actually, and fastai,  
still by default uses a reflection padding which  means if this is… I don't know, let's say this is   a 20 pixel wide thing it takes the 20 pixels next  to it and flips it over and sticks it here and  
it looks pretty good you know another one  is copy which simply takes the outside  
pixel and it's a bit more like TV you  know um…you know, much too much. Again,  
it turns out none of them really helped! plus,  you know, if anything they make it worse,  
because in the end the computer wants  to know “no this is the end of the   image there's nothing else here”,  and if you reflect it for example,  
then you're kind of creating weird spikes that  didn't exist, and the computer's got to be like,   “oh I wonder what that spike is?”, so yeah it's  a great question, and I obviously spent like a  
couple of years assuming that we should be doing  things that look more image like, but actually  
the computer likes things to be presented to  it in as straightforward a way as possible.
All right! thanks everybody! and hope  to see some of you in the walkthroughs,   and otherwise see you next time!

---

# 7

Welcome to lesson 7, the penultimate  lesson, of Practical Deep Learning for Coders  
Part 1, and today we're going to be digging  into what's inside a neural net. We've already  
seen what's inside a kind of the most basic  possible neural net, which is a sandwich of  
fully connected layers, or linear layers, and  values. And so we built that from scratch,  
but there's a lot of tweaks that we can  do, and so most of the tweaks actually  
that we probably care about are tweaking the  very first layer, or the very last layer.
So that's where we'll focus. But over  the next couple of weeks we'll look   at some of the tricks we can do inside as well.
So I'm going to do this through the lens of  
the Paddy… the Rice Paddy  competition we've been talking about,  
and we got to a point where — let's have a look…
So we created a ConvNeXt model, we tried a few  different types of basic pre-processing, we added  
Test time augmentation (TTA) and then we scaled  that up to large images, and rectangular images.
And that got us into the top  25 percent of the competition,  
so that's Part 2 of the  so-called Road to the Top series,  
which is increasingly misnamed, since  we've been presenting these notebooks,  
more and more of our students have  been passing me on the leaderboard, so,  
currently first and second place are both  people from this class: Kurian and Nick.
“Go to hell, you're in my target,  and leave my class immediately!”
And congratulations, good luck to you. So in Part 3 I'm going to show you a really  interesting trick —a very simple trick—  
What are the benefits of using larger models
for scaling up these models further.   What you'll discover if you've tried to use larger  models… so you can replace the word small with the  
word large in those architectures, and try to  train a larger model: a larger model has more  
parameters, more parameters means it can find  more tricky little features, and broadly speaking  
models with more parameters, therefore, ought  to be more accurate. The problem is that those  
activations — or more specifically —  the gradients that have to be calculated  
chews up memory on your GPU, and your GPU is not  as clever as your CPU at kind of sticking stuff  
it doesn't need right now into virtual memory  on the hard drive — when it runs out of memory:   it runs out of memory. And it also doesn't do such  a good job as your CPU at kind of shuffling things  
around to try and find memory, it just allocates  blocks of memory and it stays allocated until you  
remove them. So if you try to scale up your models  to bigger models, unless you have very expensive  
GPUs, you will run out of space, and you'll get an  error, something like “CUDA out of memory error.”  
So if that happens, first thing I  mentioned is: it's not a bad idea to  
restart your notebook because they can be  a bit tricky to recover from otherwise,  
and then I'll show you how you can  use as large a model as you like.  
Almost it's, you know, basically you'll be able to  use a x-large model on Kaggle. So, let me explain.
Now, I want… when you run something on Kaggle  — like actually on Kaggle — you're generally  
going to be on a 16 Gig GPU. And you don't have  to run stuff on Kaggle, you can run stuff on  
your home computer or Paperspace or whatever, but  sometimes you'll have — if you want to do Kaggle  
competition— sometimes you'll have to run stuff  on Kaggle because a lot of competitions are what   they call “Code Competitions” which is where the  only way to submit is from a notebook that you're  
running on Kaggle, and then a second reason to run  stuff on kaggle is that, you know, your notebooks  
will appear, you know, with the leaderboard score  on them, and so people can see which notebooks are  
actually good. And I kind of like, even in things  that aren’t code competitions, I love trying to  
be the person who's number one on the notebook  score leaderboard because that's something which,  
you know, you can't just work at NVIDIA, and use  a thousand GPUs and win a competition through  
a combination of skill and brute force. Everybody  has the same nine hour timeout to work with,  
so I think it's a good way of keeping  the, you know, things a bit more fair.  
Understanding GPU memory usage
Now… so, my home GPU has 24 Gig so I wanted to  find out what can I get away with, you know,  
in 16 Gig, and the way I did that is, I think,  a useful thing to discuss because, again, it's  
all about fast iteration. So I wanted to really  quickly find out how much memory will a model use,  
so there's a really quick hacky way I can do  that which is to say: “okay, for the training set   let's not use… (so here's the value counts  of labels, so the number of each disease…)  
let's not look at all the diseases, let's just  pick one, the smallest one, right?, and let's  
make that our training set. Our training set is  the “bacterial_panicle_blight” images, and now I  
can train a model with just 337 images without  changing anything else. Not that I care about  
that model but then I can see how much memory it  used. It's important to realize that, you know,  
each image you pass through is the same size,  each batch size is the same size, so training   for longer won't use more memory, so that'll  tell us how much memory we're going to need.
So what I then did was I then tried training  different models to see how much memory  
they… they used up. Now, what happens if we  train a model? So obviously Convnext_small  
doesn't use too much memory. So here's something  that reports the amount of GPU memory just by  
basically printing out cuda's GPU processes,  and you can see Convnext_small took up 4GB.  
And also this might be interesting  to you, if you then call Python's   garbage collection gc.collect(), and  then call Pytorch's empty_cache()  
that should basically get your GPU back to  a clean state of not using any more memory  
than it needs to, when you can start training  the next model without restarting the kernel.
What is GradientAccumulation?
So what would happen if we tried to train this  little model, and it crashed with a “cuda out  
of memory error”. What do we do? We can use a  cool little trick called gradient accumulation.  
What's gradient accumulation? So what's gradient  accumulation? Well I added this parameter to my  
train method here. That's my train method,  creates my data loaders, creates my learner,  
and then —depending on whether I'm fine  tuning or not— either fits or fine-tunes it.  
But there's one other thing it does… it  does this gradient accumulation thing.   What's that about? Well, the key step is here.  I set my batch size (so that's the number of  
images that I pass through to the GPU all at  once) to 64, which is my default. Divided by…  
(slash-slash means integer divide in Python..).  divided by this number. So if I pass 2,  
it's going to use a batch size of 32. If  I pass 4, it'll use a batch size of 16.  
Now that obviously should let me cure any memory  problems, use a smaller batch size. But the  
problem is that now, the dynamics of my training  are different, right? The smaller your batch size,  
the more volatility there is from batch to  batch. So now your learning rates are all   messed up. You don't want to be messing around  with trying to, you know, find a different set of,  
kind of, optimal parameters for every  batch size, for every architecture.  
So what we want to do is find a way to run just,  let's say accum is two, accumulate equals two,  
let's say we just want to run  32 images at a time through.   How do we make it behave as if it was 64 images?  Well, the solution to that problem is to consider  
our training loop. This is the… basically  the training loop we used from a couple of   lessons ago, the one we created manually. We  go through each (x,y) pair in the data loader.  
We calculate the loss using some  coefficients based on that (x,y) pair,   and then we call backward() on that  loss to calculate the gradients,  
and then we subtract from the coefficients, the  gradients times the learning rate. And then we  
zero out the gradient. So I've skipped a bit  of stuff like the… with torch.no_grad() thing.  
Actually, no, I don't need that because I've  got .data, no that's it. That should all work   fine. That skipped out printing the loss, that's  about it. So here is a variation of that loop…
where I do not always subtract the  gradient times the learning rate. Instead I  
go through each (x,y) pair in the data  loader, I calculate the loss, I look at  
how many images are in this batch. So initially I  start at zero, and this count is going to be 32,  
say if I've divided the batch size by 2.  And then if “count” is greater than 64,  
I do my gradient… my coefficients update.  Well it's not, so I skip back to here,
and I do this again. And if you remember there  was this interesting subtlety in Pytorch,  
which is if you call backward() again  without zeroing out the gradients,  
then it adds this set of gradients to the old  gradients. So by doing these two half size batches  
without zeroing out the gradients between  them, it's adding them up. So I'm going to   end up with the total gradient of a 64 image  batch size but passing only 32 at a time.
If I used accumulate equals four. It would go  through this four times, adding them up before  
it subtracted out the coefficients.grad  times learning rate, and zeroed it out.  
If I put in accum equals 64, it would go  through and do a single image one at a time,  
and after 64 passes through eventually,  count would be greater than 64, and we  
would do the update. So that's gradient  accumulation, right? It's… it's a very  
simple idea, right, which is that you don't  have to actually update your weights every  
loop through, for every minI batch.  You can just do it from time to time.  
But it has quite significant  implications, which I find   most people seem not to realize, which is if you  look on, like, Twitter or Reddit or whatever,  
people could say “oh I need to buy a  bigger GPU to train bigger models”.  
But they don't. They could just  use gradient accumulation, and so  
given the huge price differential between,  say, a RTX3080, and an RTX3090 Ti, huge  
price differential… the performance is not that  different. The big difference is the memory. So  
what? Just put in a bit smaller batch size, and  do gradient accumulation. So there's actually  
not that much reason to buy giant GPUs. John? JOHN: Are the results with gradient  accumulation numerically identical?
JEREMY: They're numerically identical  for this particular architecture.  
There is something called batch normalization,  which we will look at in Part 2 of the course,  
which keeps track of the moving average  of… standard deviations and averages,  
and does it in a mathematically slightly incorrect  way. As a result of which, if you've got batch  
normalization, then it could… it basically  will introduce more volatility. Which is not  
necessarily a bad thing, but because it's not  mathematically identical, you won't necessarily   get the same results. Convnext doesn't use batch  normalization, so it is the same. And in fact,  
a lot of the models people want to use, really  big versions of which, is NLP ones, Transformers,  
tend not to use batch normalization, but instead  they use something called layer normalization,  
which, yeah, doesn't have the same issue. I  think that's probably fair to say. I haven't  
thought about it that deeply. In practice I found  adding gradient accumulation for Convnext has not  
caused any issues for me, and I don't have  to change any parameters when I do it.
Any other questions on the forum, John? JOHN: Tamori asking shouldn't it  be count>=64 if bs=64, I haven't…
JEREMY: No, I don't think so, oh yeah, is that…  so we start at zero, then it's gonna be 32,  
then it's gonna be… yeah yeah, probably yeah, you  can probably tell I didn't actually run this code.
JOHN: Madav is asking: does this mean that   lr_find() is based on the batch  size set during the data block?
JEREMY: Yeah, so lr_find() just  uses your data loaders batch size.
JOHN: Edward is asking: why do we need  gradient accumulation rather than just  
using a smaller batch size, and follows up  with how would we pick a good batch size? JEREMY: Well just, if you use a smaller  batch size… here's the thing right,  
different architectures have different amounts of  memory, you know, which they which they take up,  
and so you'll end up with different  batch sizes for different architectures,  
which is not necessarily a bad thing but each of  them is going to then need a different learning   rate, and maybe even different weight decay  or whatever, like, the kind of… the settings  
thats working really well for batch size 64 won't  necessarily work really well for batch size 32.  
And you know, you want to be able to  experiment as easily, and quickly as possible.  
I think the second part of the question was “how  do you pick a optimal batch size?” Honestly,  
the standard approach is to  pick the largest one you can,   just because it's faster that way – you're  getting more parallel processing going on.  
Although to be honest I quite often use batch  sizes that are quite a bit smaller than I need  
because quite often it doesn't  make that much difference,   but yeah, the rule of thumb would be, you know,  pick a batch size that fits in your GPU, and for  
performance reasons I think it's generally a good  idea to have it be a multiple of eight. Everybody  
seems to always use powers of two, I don't  know, like, I don't think it actually matters. JOHN: And look there's one other,  just a clarification or a check,  
if the learning rate should be  scaled according to the batch size? JEREMY: Yeah so generally speaking the rule of  thumb is that if you divide the batch size by two,  
you divide the learning rate by two, but  unfortunately it's not quite perfect.
Did you have a question Nick? If  you do you can, okay cool yeah. JOHN: Yeah no, that's us all  caught up, thanks Jeremy.
JEREMY: Good questions thank you.
So gradient accumulation in fastai  is very straightforward, you just  
divide the batch size by however much you  want to divide it by, and then add a… you   got something called a callback, and a callback is  something which changes the way the model trains.  
This callback is called GradientAccumulation, and  you pass in the effective batch size you want,  
and then you say, when you create the learner  you say, these are the callbacks I want,   and so it's going to pass in GradientAccumulation  callback. So it's going to only  
update the weights once it's got 64 images.
So if we pass in accum=1 it won't  do any gradient accumulation,   and that uses 4GB. If we use accum=2 about 3GB,  accum=4 about 2.5GB, and generally the bigger  
the model the closer you'll get to a kind of  a linear scaling because models have a kind of  
a bit of overhead that they have anyway.  So what I then did, was I just went through  
all the different models I wanted to try. So I  wanted to try convnext_large at a 320 by 240,  
vit_large, swinv2_large, swin_large, and on each  of these I just tried running it with accum=1, and  
actually every single time for all of these I got  an “out of memory” error. And then I tried each of   them independently with accum=2, and it turns out  that all of these worked with accum=2, and it only  
took me 12 seconds each time, so that was a very  quick thing for me. Then okay, I now know how to  
train all of these models on a 16GB card. So I  can check here, they're all in less than 16GB.  
So then I just created a little dictionary  of all the architectures I wanted,  
and for each architecture all of the resize  methods I wanted, and final sizes I wanted.  
Now these models vit, swinv2, and  swin, are all Transformers models,  
which means that, well, most Transformers models,  nearly all of them, have a fixed size – this one's  
224, this one's 192, this one's 224. So I have  to make sure that my final sizes are square,  
of the size required, otherwise I get an error.   There are… there is a way of working around this  but I haven't experimented with it enough to know  
when when it works well, and when it doesn't,  so we'll probably come back to that in Part 2.  
So for now we're just going to use  the size that they ask us to use. So with this dictionary of architectures, and for  each architecture, kind of pre-processing details,  
How to run all the models with specifications
we switch the training path  back to using all of our images,   and then we can loop through each architecture,  
and loop through each item transfer– transforms  and sizes, and train the model, and then  
the training script, if you're  fine-tuning, returns the tta predictions.  
So I append all those tta predictions,  for each model for each type, into a list,  
and after each one it's a good idea to do this  garbage collection, and empty cache that… because  
otherwise I find what happens is your GPU memory  kind of, I don't know, I think it gets fragmented  
or something, and after a while it runs out  of memory even when you thought it wouldn't.   So this way you can really do as much as you like  without running out of memory. So they all train,  
train, train, train, and one key thing  to note here, is that in my train script,  
my data loaders does not have the seed= parameter,  so I'm using a different training set every time.
So that means that for  
each of these different runs they're using also  different validation sets, so they're not directly   comparable, but you can kind of see they're all  doing pretty well, 2.1%, 2.3%, 1.7%, and so forth.  
So why am I using different training  and validation sets for each of these?   That's because I want to ensemble  them, so I'm going to use bagging,  
Ensembling
which is I am going to take the average of  their predictions. Now I mean really, when  
we talked about random forest bagging, we were  taking the average of, like, intentionally weak  
models. These are not intentionally weak models,  they're meant to be good models, but they're all   different – they're using different architectures,  and different pre-processing approaches, and so  
in general we would hope that these different  approaches… some might work well for some images,   and some might work well for other images. And  so when we average them out, hopefully we'll get  
a good blend of, kind of, different ideas,  which is kind of what you want in bagging.
So we can stack up that list of different… of all  the different probabilities, and take their mean,  
and so that's going to give us  3469 predictions, that's our   that's our test set size, and each one has 10  probabilities, the probability of each disease.  
And so then we can use argmax() to find  which probability index is the highest,  
so that's going to give us our list of indexes.   So this is basically the same steps as we  used before to create our CSV submission file.
So at the time of creating this analysis that  got me to the top of the leaderboard, and in fact  
these are my four submissions, and  you can see each one got better.  
Now you're not always going to get  this nice monotonic improvement,   right, but you want to be trying  to submit something every day,  
to kind of, like, try out something new, right,  and the more you practice the more you'll get a  
good intuition of what's going to help, right. So  partly I'm showing you this to say, it's not like  
purely random as to whether things work or don't,  once you've been doing this for a while, you know,   you will generally be improving things most of the  time. So as you can see from the descriptions my  
first submission was our convnext_small for 12  epochs with TTA. And then ensemble of convnext,  
so it's basically this exact same thing  but just retraining a few with different   training subsets. And then this is the same thing  again, this is the thing we just saw, basically  
the ensemble of large models with TTA. And then  the last one was something I skipped over, which  
was I… the VIT models were the best in my testing,  so I basically weighted them as double in the  
ensemble – pretty unscientific but again it gave  it another boost, and so that was that was it.
all right, John! JOHN: Yes, thanks Jeremy. So in no particular  order Kurian is asking, would trying out cross  
validation with k-folds with the same architecture  makes sense as an ensembling of models?
JEREMY: Yeah, so a popular thing is  to do k-fold cross-validation. So  
k-fold cross-validation is something  very very similar to what I've done here. So what I've done here is I've trained a bunch of  models with different training sets, each one is  
a different random 80% of the data. Five-fold  cross-validation does something as similar,  
but what it says is rather than picking, like say  five samples out with different random subsets,  
in fact, instead, first like… do all except for  the first 20% of the data, and then all but the  
second 20%, and then all but the third, and so  forth, and so you end up with five subsets each  
of which have non-overlapping validation sets,  and then you'll ensemble those. You know in theory  
maybe that could be slightly better because  you're kind of guaranteed that every row is…  
appears… four times, you know, effectively.  It also has a benefit that you could average  
those five validation sets because there's no kind  of overlap between them to get a cross validation.  
Personally, I generally don't bother, and  the reason I don't is because this way  
I can add and remove models very  easily. I don't, you know… I can just,  
you know… add another architecture and whatever  to my ensemble without trying to find a   different overlapping non-overlapping  subset. So yeah, cross-validation is  
therefore something that I use probably less  than most people or almost… or almost never.
JOHN: Awesome, thank you. Are there  any… just coming back to gradient  
accumulation… any other kind of drawbacks or  potential gotchas with gradient accumulation? JEREMY: No, not really! Yeah, like, amazingly  it doesn't even really slow things down much,  
you know, going from a batch  size of 64 to a batch size of 32.   By definition you had to do it because your GPU  is full so you're obviously giving a lot of data.  
So it's probably going to be using its processing  speed pretty effectively, so yeah, no it's just…  
it's just a good technique that… we should all be  buying cheaper graphics cards with less memory in  
them, and using you know, have like… I don't know  the prices, I suspect it's like you could probably  
buy like two 3080s for the price of one 3090Ti  or something, that would be a very good deal.
JOHN: Yes, clearly you're  not on the Nvidia payroll.   So look this is a good segue then, we did have  a question about sort of GPU recommendations,  
and there's been a bit of chat on that  as well. (JEREMY: I bet) So any… any,   you know, commentary… any additional  commentary around GPU recommendations.
JEREMY: No, not really. I mean obviously at the  moment Nvidia is the only game in town, you know.  
If you buy… if you trying to use a you know Apple  M1 or M2 or an AMD card you're basically in for  
a world of pain in terms of compatibility and  stuff, and unoptimized libraries, and whatever.  
The Nvidia consumer cards, so the ones that start  with RTX are much cheaper but are just as good  
as the expensive enterprise cards. So you might  be wondering why anybody would buy the expensive  
enterprise cards, and the reason is that there's a  licensing issue that Nvidia will not allow you to  
use an RTX consumer card in a data center – which  is also why cloud computing is more expensive than  
they, kind of, ought to be, because everybody  selling cloud computing GPUs is selling  
these cards that are like, I can’t remember, I  think they're like three times more expensive   for kind of the same features. So yeah, if you do  get serious about deep learning to the point that  
you're prepared to invest, you know, a few  days in administering a box, and you know,  
I guess depending on prices, hopefully will start  to come down, but currently a thousand or two   thousand dollars, and buying a GPU then you know,  that'll probably pay you back pretty quickly.
JOHN: Great, thank you. Let's see,  another one has come in. If you have a…  
back on models, not hardware… if you have  a well functioning but large model, can it   make sense to train a smaller model to produce  the same final activations as the larger model?
JEREMY: Oh yeah, absolutely. I'm not sure  we'll get into that this time around but  
yeah we'll cover that in Part 2, I think, but  yeah basically there's teacher/student models,  
and model distillation, which  broadly speaking there are ways to  
make inference faster by training small  models that work the same way as large models.
JOHN: Great thank you, all caught up. JEREMY: All right, so… that is the actual real  end of Road to the Top, because beyond that  
we don't actually cover how to get closer to  the top – you'd have to ask Kurian to share his   techniques to find out that, or Nick, to get the  second place from the top. Part 4 is actually  
something that I think is very useful to know  about for learning, and it's going to teach   us a whole lot about how the last layer  of a neural net works. And specifically,  
what we're going to try to do is we're going to  try to build a model that doesn't just predict  
the disease but also predicts the type of rice.   So how would you do that? So here's the  Data Loader we're going to try to build –  
it's going to be something that for each image  it tells us the disease, and the type of rice.  
I say disease, sometimes normal, I  guess some of them are not diseased.
So to build a model that can predict two  things, the first thing is you're going   to need data loaders that have two dependent  variables, and that is shockingly easy to do  
in fastai thanks to the DataBlock. So we've seen  the DataBlock before. We haven't been using it  
for the Paddy competition so far because  we haven't needed it – we could just use   ImageDataLoader.from_folder(). So that's like the  highest level API, the simplest API. If we go down  
a level deeper into the DataBlock we have a lot  more flexibility. So if you've been following the  
walkthroughs you'll know that as I built this  the first thing I actually did was to simply   replicate the previous notebook but replace the  ImageDataloader.from_folder() with a DataBlock to  
try to do, first of all, exactly the same thing,  and then I added the second dependent variable.
So if we look at the previous  ImageDataLoader.from_folder() thingy,  
here it is. We were passing in some item  transforms, and some batch transforms,  
and we had something saying what  percentage should be the validation set
So in a DataBlock if you remember we have to  pass in a “blocks” argument saying what kind  
of data is the independent variable, and what  is the dependent variable. So to replicate what  
we had before we would just pass an ImageBlock  comma CategoryBlock, because we've got an image   as our independent variable, and a category,  one type of rice, is the dependent variable.
So the new thing I'm going to show you here,is  that you don't have to “only put in two things.”   You can put in as many as you like, so if you put  in three things we're going to generate one image,  
and two categories. Now fastai, if you're saying  I want three things… fastai doesn't know which  
of those is the independent variable,  and which is the dependent variable,   so the next thing you have to tell it is how  many inputs are there, “number of inputs,”  
And so here I've said there's one  input, so that means this is the input,   and therefore by definition two categories will  be the output because remember we're trying to  
predict two things the type of rice, and the  disease. Okay this is the same as what we've  
seen before, to find out… to get our list of  items we'll call get_image_files(). Now here  
is something we haven't seen before – get_y is  our labeling function. Normally we pass to get_y  
a single thing such as the parent_label function  which looks at the name of the parent directory,  
which remembers how these images are structured,  and that would tell us the label. But get_y  
can also take an array, and in this case we want  two different labels. One is the name of the  
parent directory, because that's the disease. The  second is the variety. So what is get_variety()?
get_variety() is a function. So let  me explain how this function works.  
So we can create a data frame containing  our trainings... our training data that   came from Kaggle. So for each image it  tells us the disease, and the variety.
And what I did is something I haven't shown  before. In pandas you can set one column to   be the index, and when you do that,  in this case image_id, it makes this  
series... this... sorry… this data frame, kind  of like a dictionary. I can index into it by  
saying tell me the row for this image, and to do  that you use the “loc” attribute, the location.  
So we want in the data frame, the location of  
this image, and then you can also say optionally  what column you want – this column. And so  
here's this image, and here's this column,  and as you can see it returns that thing.  
So hopefully now you can see it's pretty  easy for us to create a function that takes  
a row... sorry, a path, and returns  the location in the data frame  
of the name of that file, because remember these  are the names of files, for the variety column.
So that's our second get_y   okay, and then we've seen this before,  randomly split the data into the 20% and 80%.
And so we could just switch them  all to 192 just for this example,   and then use data augmentation to get us down  to 128 square images just for this example.  
And so that's what we get when we say  show batch. We get what we just discussed.
Multi-target models
So now we need a model that predicts two things.  
How do we create a model that predicts two  things? Well, the key thing to realize is we never  
actually had a model that predicts two things.  We had a model that predicts ten things, before.  
The ten things we predicted is the probability  of each disease. So we don't actually now want  
a model that predicts two things, we  want a model that predicts 20 things:   the probability of each of the 10 diseases, and  the probability of each of the 10 varieties.
So, how could we do that? Well let's first  of all try to just create the same disease  
model we had before with our new data loader. So  this is going to be reasonably straightforward.  
The key thing to know is that since we told  fastai that there's one input, and therefore  
by definition there's two outputs, it's going to  pass to our metrics, and to our loss functions,  
three things instead of two: the predictions  from the model, and the disease, and the variety.
So if we're gonna... so we can't just use  error rate as our metric anymore because   error rate takes two things. Instead we have  to create a function that takes three things,  
and return error rate of the two things we  want, which is the predictions from the model,  
and the disease, okay? So this is  predictions of the model, this is the target.   So that's actually all we need to do to define a  metric that's going to work with our new dataset…  
with a new dataloader, and this is not going  to actually tell us anything about variety,   first we're just going to try to replicate  something that can do just disease.  
So when we create our learner we’ll  pass in this new disease error function.
Okay, so we're halfway there. The  other thing we're going to need   is to change our loss function. Now we never  actually talked about what loss function to use,  
and that's because vision_learner guessed what  loss function to use. vision_learner saw that our  
dependent variable was a single category, and it  knows the best loss function that's probably going  
to be the case for things with a single category,  and it knows how big the category is. So it just  
didn't bother us at all. It just said okay “I'll  figure it out for you”. So the only time we've  
provided our own loss function is when we were  kind of doing linear models and neural nets from  
scratch, and we did, I think, mean-squared-error.  We might also have done mean-absolute-error.
Neither of those work when the dependent variable   is a category. Now how would you use  mean-squared-error or mean-absolute-error  
to say, how close were these 10 probability  predictions to this one correct answer.
So in this case we have to use a  different loss function. We have   to use something called cross entropy loss,  and this is actually the loss function that  
fastai picked for us before without us  knowing. But now that we are having to   pick it out manually I'm going to explain to  you exactly what cross-entropy loss does, okay?
What does `F.cross_entropy` do
And you know these details are very important  indeed. Like... remember I said at the start  
of this class, the stuff that happens in the  middle of the model you're not going to have   to care about much in your life, if ever, but  the stuff that happens in the first layer, and  
the last layer including the loss function that  sits between the last layer and the loss, you're   going to have to care about a lot, right? This  stuff comes up all the time, so you definitely  
want to know about cross-entropy loss, and so  I'm going to explain it using a spreadsheet.
This spreadsheet's in the course repo, and so  let's say you are predicting something like  
a... kind of a... mini imagenet thing where you're  trying to predict whether something... an image,  
is a cat, a dog, a plane, a fish or a building.  So you set up some model, whatever it is,  
a convnext model, or a just a big bunch of  linear layers connected up, or whatever, and  
initially you've got some random  weights, and it spits out at the end,  
five predictions, right? So remember, to predict  something with five categories, your model will  
spit out five probabilities. Now it doesn't  initially spit out probabilities. There's nothing  
making them probabilities, it  just spits out five numbers;   could be negative, could be positive,  okay? So here's the output of the model.
So what we want to do is: we want  to convert these into probabilities.  
And so we do that in two steps.  The first thing we do is we go…  
EXP, that's e-to-the-power-of. We go  e-to-the-power-of each of those things,  
like so... okay? And so here's the mathematical  formula we're using. This is called the softmax,  
what we're working through. We're going  to go through each of the categories.  
So these are our five categories –  so here k is five. We've got to go   through each of our categories, and  we're going to go e-to-the-power-of  
the output, so zj is the output for the j-th  category. So here's that, and then we're going to  
sum them all together. Here it is... sum up  together, okay? So this is the denominator,  
and then the numerator is just e-to-the-power-of  the thing we care about. So this row.
So the numerator is e-to-the-power-of cat on  this row, e-to-the-power-of dog on this row,  
and so forth. Now if you think about it, since the  denominator adds up all the e-to-the-power-ofs,  
then when we do each one divided by the sum, that  means the sum of these will equal 1 by definition,  
right? And so now we have things that can be  treated as probabilities. They're all numbers  
between 0 and 1. Numbers that were bigger in  the output will be bigger here. But there's  
something else interesting, which is, because we  did e-to-the-power-of, it means that the bigger  
numbers will be, like, pushed up to numbers  closer to one, like we're saying, like, “oh   really try to pick one thing” as having most of  the probability, because we are trying to predict,  
you know, one thing. We're trying to predict  which one is it, and so this is called softmax.
So sometimes you'll see people complaining about  the fact that their model, which they said...  
let's say, is it a teddy bear or a grizzly bear or  a black bear, and they feed it a picture of a cat,  
and they say “oh the model's wrong because  it predicted grizzly bear but it's not a   grizzly bear“. As you can see there's no  way for this to predict anything other than  
the categories we're giving it – we're forcing  it to that. Now we don't... if you want that,  
When do you use softmax and when not to?
like, there's something else you could do which  is you could actually have them not add up to one,  
right? You could instead have something which  simply says: what's the probability it's a cat,   what's the probability it's a dog, what's the  probability it’s a plane, totally separately  
they could add up to less than one, and in that  situation you can cert... you know... or or more  
than one in which case you could have like more  than one thing being true or zero things being   true. But in this particular case where we want  to predict one and one thing only, we use softmax.
Cross_entropy loss
The first part of the cross entropy formula...
The first part of the cross entry formula...  in fact let's look it up, nn.CrossEntropyLoss.
The first part of what cross-entropy loss  in Pytorch does is to calculate the softmax.  
It's actually the log of the softmax but don't  worry about that too much, it's just a... slightly   faster to do the log, okay? So now for each  one of our five things we've got a probability.
The next step is the actual cross-entropy  calculation, which is we take our five things,  
we've got our five probabilities, and then we've  got our actuals. Now the truth is the actual, you  
know, the five things would have indices, right?  Zero, one, two, three or four, and the actual  
turned out to be the number one, but what we tend  to do is we think of it as being one-hot-encoded,  
which is we put a one next to the thing for  which it's true, and a zero everywhere else.  
And so now we can compare these five numbers to  these five numbers, and we would expect to have  
a smaller loss if the softmax was high where the  actual is high. And so here's how we calculate...  
this is the formula... the cross-entropy loss.
We sum up… (they switch to M this time for some  reason, but the same thing…) we sum up across   the five categories so M is 5, and for each one we  multiply the actual target value, so that's zero…  
so here it is here the actual target value,  
and we multiply that by the log of  the predicted probability… the log of  
(red) the predicted probability.  And so, of course, for four of these  
that value is zero, because see  here: y-j equals zero, by definition,  
for all but one of them, because it’s one  hot encoded. So for the one that it's not  
we've got our actual times the log softmax,  okay, and so now actually you can see why  
pytorch prefers to use log softmax because then it  kind of skips over having to do this log at all.
So this equation looks slightly frightening but  when you think about it all it's actually doing  
is: it's finding the probability  for the one that is one,   and taking its log, right. It's kind of weird  doing it as a sum but in math it could be a  
little bit tricky to kind of say, oh look this up  in an array, which is basically what it's doing,   but yeah basically, at least in this case for  a single result with soft max this is all it's  
doing, it's finding the 0.87 where it's 1 for,  and taking the log, and then finally negative.
So that is what cross-entropy loss does.  
How to calculate binary-cross-entropy
We add that together for every row.  
So here's what it looks like if we add it  together over every row, right, so N is the  
number of rows. And here's a special case, this  is called “binary cross-entropy”. What happens  
if we're not predicting which of five things  it is but we're just predicting “is it a cat?”  
So in that case if you look at this approach  you end up with this formula, which it's  
exactly… this is identical to this formula but  in for just two cases, which is you've either:  
you either are a cat; or you're not a cat,  right, and so if you're not-a-cat, it's one minus  
you-are-a-cat, and same with the probability  you've got the probability you-are-a-cat,  
and then not-a-cat is one minus that. So here's  this special case of binary cross entropy,  
and now our rows represent rows of data, okay,  so each one of these is a different image,   a different prediction, and so for each  one I'm just predicting are-you-a-cat,  
and this is the actual, and so the actual  are-you-not-a-cat is just one minus that.  
And so then these are the predictions  that came out of the model,   again we can use soft max or it's  binary equivalent, and so that will  
give you a prediction that you're-a-cat, and the  prediction that it's not-a-cat is one minus that.
And so here is:   each of the part “y-i” times log of “p-y-i”,  and here is…(why did I subtract that's weird,  
oh because I've got minus of both, so I  just do it this way, avoids parentheses…)  
yeah, minus the are-you-not-a-cat times  the log of the prediction value not-a-cat,  
and then we can add those together, and so that   would be the binary cross-entropy loss of  this dataset of five cat or not-cat images.
Two versions of cross-entropy in pytorch
Now if you've got an eagle  eye, you may have noticed that  
I am currently looking at the documentation  for something called “nn.CrossEntropyLoss”  
but over here I had something  called “F.cross_entropy()”.   Basically it turns out that all of the loss  functions in pytorch have two versions – there's  
a version which is a class, this is a class,  which you can instantiate passing in various  
tweaks you might want, and there's also  a version which is just a function,  
and so if you don't need any of these tweaks  you can just use the function. The functions  
live in a… I can’t even remember what the  sub module called, I think it might be like  
torch.nn.functional but everybody including the  pytorch official docs just calls it capital-F,  
so that's what this capital-F refers to. So our loss, if we just care about disease  (we're going to be passed the three things)  
we're just going to calculate cross_entropy  on our input versus disease. All right  
so that's all fine… we passed… so now when  we create a vision learner you can't rely on  
fastaI to know what loss function to use, because  we've got multiple targets, so you have to say:  
this is the loss function I want to use, this  is the metrics I want to use. And the other   thing you can't rely on is that fastaI no  longer knows how many activations to create,  
because again it… there's more than one target,  so you have to say the number of outputs to create  
at the last layer is 10. So this is just  saying what's the size of the last matrix.
And once we've done that we can train it, and we  get, you know, basically the same kind of result  
as we always get, because this model at this  point is identical to our previous convnext_small  
model – we've just done it in a slightly more  roundabout way. So finally, before our break,  
How to create a learner for prediction two targets
I'll show you how to expand this  now into a multi-target model.   And the trick is actually very simple, and you  might have almost got the idea of it when I talked  
about it earlier – our vision learner now requires  twenty outputs – we now need that last matrix  
to have to produce twenty activations not ten.  Ten of those activations are going to predict  
the disease, and ten of the activations  are going to predict the variety. So you  
might be then asking, like well, how does the  model know what it's meant to be predicting,   and the answer is with the loss function,  you're going to have to tell it.
So for example disease_loss –  remember it's going to get the input,   the disease, and the variety – this  is now going to have 20 columns in.  
So we're just going to decide, all right, we're  just going to decide the first 10 columns,   we're going to decide are the  prediction of what the disease is,  
which is the probability of each disease.  So we can now pass to cross_entropy  
the first 10 columns, and the disease target.  So the way you read this colon means every row,  
and then colon 10 means every column up to the  10th. So these are the first 10 columns, and that  
will… that's a loss function that just works on  predicting disease using the first ten columns.
For variety, we'll use cross_entropy  loss with the target of variety,   and this time we'll use the second 10  columns, so here's column ten onwards.  
So then the overall loss function is the sum of  those two things disease_loss plus variety_loss.  
And that's actually it! That's all the model  needs to basically… it's now going to… if  
you kind of think through the manual neural nets  we've created, this loss function will be reduced  
when the first 10 columns are doing good  job of predicting the disease probabilities   and the second 10 columns are doing a good  job of predicting the variety probabilities  
and therefore the gradients will point in an  appropriate direction that the coefficients   will ~(be getter and…) get better and better  at using those columns for those purposes.
It would be nice to see the error rate  as well for each of disease and variety,   so we can call error_rate passing in the first  10 columns and disease, and then variety,  
the second 10 columns and variety. And we may  as well also add to the metrics the losses.  
And so now when we create our learner  we're going to pass in as the loss function   the combined_loss – and as the metrics, our list  of all the metrics and n_out=20 and now look what  
happens when we train! As well as telling us the  overall train and valid loss, it also tells us the  
disease and variety error and the disease and  variety loss and you can see our disease error  
is getting down to a similar level as it was  before. It's slightly less good but it's similar.  
It's not surprising it's slightly less good  because we've only given it the same number  
of epochs and we're now asking it to try to do  more stuff, which is to learn to recognize what  
the rice variety looks like, and also learns  to recognize what the disease looks like.
Here's the counterintuitive thing  though if we train it for longer  
it may well turn out that this model  which is trying to predict two things  
actually gets better at predicting disease than  our disease specific model. Why is that?... like  
that sounds weird, right, because we're trying to  do more stuff, [and] that's the same size model.  
Well the reason is that quite often it'll turn out  that the kinds of features that help you recognize  
a variety of rice are also useful for recognizing  the disease. You know, maybe there are certain  
textures, right, or maybe some diseases  impact different varieties different ways,  
so it'd be really helpful to know what variety  it was. So I haven't tried training this for  
a long time and I don't know the answer is… In  this particular case does a multi-target model  
do better than a single target model at predicting  disease, but I just want to let you know sometimes  
it does, right. So for example a few years ago  there was a Kaggle competition for recognizing  
the kinds of fish on a boat, and I remember  we ended up doing a multi-target model  
where we tried to predict a second thing… (I can't  even remember what it was, maybe it was a type of   boat or something…) and it definitely turned  out in that Kaggle competition that predicting  
two things helped you predict the type of fish  better than predicting just the type of fish.  
So there's at least, you know, there's two reasons  to learn about multi-target models: one is that  
sometimes you just want to be able to predict  more than one thing, so this is useful;   and the second is, sometimes this will actually  be better at predicting just one thing,  
than a just one thing model. And of course  the third reason is it really forced us to  
dig quite deeply into these loss functions and  activations in a way we haven't quite done before
So it's okay, it's absolutely okay, if this  is confusing. The way to make it not confusing  
is… well… the first thing I do is, like, go  back to our earlier models where we did stuff   by hand, on like the titanic dataset and built  our own architectures, and maybe you could try  
to build a model that predicts two things in the  titanic dataset. Maybe you could try to predict  
both sex & survival or something like that,  or class & survival, because that's going to,  
kind of, force you to look at it on very small  datasets. And then the other thing I'd say is   run this notebook and really experiment at  trying to see what kind of outputs you get,  
like actually look at the inputs and look at the  outputs and look at the data loaders and so forth. All right let's have a six minute break, so  I'll see you back here at ten past seven.
Okay, welcome back. Before I continue, I  very rudely forgot to mention this very nice  
equation image here is from an article by  Chris Said called “Things that confused me  
about cross-entropy.” It's a very good article,  so I recommend you check it out if you want to  
go a bit deeper there. There's a  link to it inside the spreadsheet.
So the next notebook we're going to be looking  at is this one called Collaborative Filtering   Deep Dive, and this is going to cover our  last of the four major application areas,  
Collaborative filtering deep dive
collaborative filtering.
And this is actually the first time I'm going  to be presenting a chapter of the book largely  
without variation because this is one where  I looked back at the chapter and I was like,  
“oh I can't think of any way to improve this.” So  I thought I'll just leave it as is, but we have  
put the whole chapter up on Kaggle, so that's  for the way I'm going to be showing it to you.
And so we're going to be looking at a  dataset called the MovieLens dataset,  
which is a dataset of movie ratings,   and we're going to grab a smaller version  of it, 100 000 record version of it,
and it comes as a csv file which we can read in,   well, it's not really a csv file, it's a  tsv file, this here means a tab in Python.
These are the names of the columns. So here's  what it looks like. It's got a user, a movie,  
a rating, and a timestamp. We're not going to  use the timestamp at all. So basically three  
columns we care about. This is a user id, so  maybe 196 is Jeremy, and maybe 186 is Rachel,  
and 22 is John, I don't know. Maybe this movie  is Return Of The Jedi, and this one's Casablanca,  
this one's LA Confidential. And then this rating  says how did Jeremy feel about Return Of The Jedi,  
he gave it a three out of five. That's how we  can read this dataset. This kind of data is very  
common. Anytime you've got a  user and a product or service,  
and you might not even have ratings,  maybe just the fact that they   bought that product, you could have a similar  table with zeros and ones. So, for example  
Radek, who's in the audience here, is now at  Nvidia doing, like, basically just this, right,   recommendation systems. So recommendation  systems, you know, it's a huge industry,  
and so what we're learning today is, you  know, a really key foundation of it. Um…
So these are the first few rows. This is not a  particularly great way to see it. I prefer to kind   of cross tabulate it, like that, like this. This  is the same information. So for each movie, for  
each user, here's the rating. So user 212 never  watched movie 49. Now if you're wondering, uh…
why there's so few empty cells here, I actually  grabbed the most watched movies and the most movie  
watching users for this particular sample matrix.  So that's why it's particularly full. So yeah,  
so this is what kind of a collaborative filtering  dataset looks like when we cross tabulate it.
So how do we fill in this gap? So maybe  user 212 is Nick, and review 49… what's  
a movie you haven't seen, Nick, and you'd  quite like to? Maybe, not sure about it,  
the new Elvis movie, Bez Luhmann? Good choice.  Australian director, filmed in Queensland. Yeah,  
okay, so that's movie two… that's movie number  49. So is Nick gonna like the new Elvis movie?  
Well, to figure this out, what we could do,  ideally, would like to know: for each movie…  
what kind of movie is it? Like, what are the  kind of features of it. Is it like actiony,  
science fictiony, dialogue  driven, critical acclaimed,   you know. So let's say, for example, we were  trying to look at The Last Skywalker –maybe  
that was the movie the… Nick's wondering  about watching. And so if we like, had,  
three categories, being science fiction, action,  or kind of classic old movies, would say The Last  
Skywalker is very science fiction. Let's  see this is from like negative one to one,  
pretty action, definitely not an  old classic, or at least not yet.
And so then, maybe we then could say,  like, okay, well, maybe like Nick’s tastes  
in movies are that he really likes science  fiction, quite likes action movies,  
and doesn't really like old classics, right.  So then we could, kind of, like, match these up  
to see how much we think this  user might like this movie  
to calculate the match, we could just  multiply the corresponding values  
user1 times The Last Skywalker, and add  them up, point nine (0.9) times point nine   eight (0.98), plus point eight (0.8) times point  nine (0.9), plus negative point six (-0.6) times  
negative point nine (-0.9). That's going to give  us a pretty high number, right, with a maximum   of three. So that would suggest Nick probably  would like The Last Skywalker. On the other hand,  
the movie Casablanca we would say  definitely not very science fiction,  
not really very action, definitely very old  classic. So then we'd do exactly the same  
calculation and get this negative result here.  So you probably wouldn't like Casablanca. This  
thing here, when we multiply the corresponding  parts of a vector together and add them up,  
is called a dot product in math. So this is  the dot product of the user's preferences and  
the type of movie. Now the problem  is, we weren't given that information.  
We know nothing about these users or about  the movies. So what are we going to do? 
We want to try to create these  factors without knowing ahead of time  
what they are. We wouldn't even know what  factors to create, what are other things   that really matters when it says people  decide what movies they want to watch?
What are latent factors?
What we can do is we can create things called  latent factors. Latent factors is this weird  
idea that we can say: I don't know what  things about movies matter to people,  
but there's probably something and let's just  try, like, using SGD to find them. And we can  
do it in everybody's favorite mathematical  optimization software: Microsoft Excel.
So here is that table.
And, what we can do –let's head over here  actually– here's that table. So, what we  
could do is we could say: for each of those  movies –so let's say for movie 27– let's assume  
there are five latent factors –I don't know what  they're for–, they're just five latent factors,  
we'll figure them out later. And for now I  certainly don't know what the value of those  
five latent factors for movie 27, so we're going  to just chuck a little random numbers in them,  
and we're going to do the same thing for movie 49  –pick another five random numbers– and the same   thing for movie 57 –pick another five numbers. And  you might not be surprised to hear we're going to  
do the same thing for each user, so for user 14:  we're going to pick five random numbers for them,  
and for user 29: we'll pick  five random numbers for them,   and so the idea is that this number here 0.19  is saying –if it was true– that user id 14  
feels not very strongly about the fact that for  movie 27 has a value of 0.71. So therefore in here  
we do the dot product. The details of why don't  matter too much but, well, actually you can  
figure this out from what we've said so far: if  you go back to our definition of matrix product  
you might notice that the matrix product of a row  with a column is the same thing as a dot product  
and so here in excel I have a row and a column  so, therefore I say matrix multiply that by that:  
that gives us the dot product. So here's the dot  product of that by that –or the matrix multiply,  
given that they're row and column. The only other  slight quirk here is that if the actual rating  
Dot product model
is zero –is empty– I'm just going to leave it  blank, I'm going to set it to zero actually.
So here is everybody's rating,  predicted rating of movies.  
I say predicted –of course these are currently  random numbers so they are terrible predictions.  
But when we have some way to predict things  and we start with terrible random predictions,   we know how to make them better, don't  we?, we use static gradient descent.  
Now to do that we're going to need a  loss function, so that's easy enough,  
we can just calculate the sum of x  minus y squared divided by the count,  
that is the mean squared error –and if we take the  square root, that is the root mean squared error–  
so here is the root mean squared error, in Excel,  between these predictions and these actuals.
And so now that we have a loss function,  we can optimize it. Data solver,  
set objective (this one here) by changing  cells (these ones here) and (these ones here).  
Solve. Okay, and initially our loss is 2.81 –so  we hope it's going to go down– and as it solves  
–not a great choice of background color– but  it says 0.68 so this number is going down. So  
this is using… actually an Excel it's not quite  using stochastic gradient descent because excel  
doesn't know how to calculate gradients,  there are actually optimization techniques   that don't need gradients: they calculate them  numerically as they go but that's a minor quirk.  
One thing you'll notice is  it's doing it very, very slowly   –there's not much data here and it's still going–  one reason for that is that… it's because it's  
not using gradients it's much slower and  the second is Excel is much slower than  
Pytorch. Anyway, it's come up with an answer,  and look at that: it's got to 0.42. So it's  
got a pretty good prediction and so, we can  kind of get a sense of this, for example,  
looking at the last three,  
movie user 14 likes, dislikes, likes. Let's see  somebody else like that. Here's somebody else,  
this person likes, dislikes, likes. So, based  on our kind of approach we're saying: okay,  
since they have the same feeling about these three  movies, maybe they'll feel the same about these   three movies. So this person likes all three  of those movies and this person likes two out  
of three of them so, you know, you kind of… this  is the idea, right?, as if somebody says to you:   “I like this movie, this movie, this movie” and  you're like: “oh, they like those movies too” what  
other movies do you like? and they'll say: “oh,  how about this?” There's a chance, good chance,   that you're going to like the same thing. That's  the basis of collaborative filtering, okay, it's…  
and mathematically we call this matrix completion.  So this matrix is missing values, we just want to…  
complete them. So the core of collaborative  filtering is, it's a matrix completion exercise.
Can you grab a microphone?
NICK:   Is that better? Okay, my question was,  is, with the dot products, right?, so,  
if we think about the math of that for a minute  is, yeah, if we think about the cosine of the   angle between the two vectors that's going  to roughly approximate the correlation,  
is that essentially what's going  on here in one sense? with… JEREMY: Okay so is the cosine of the  angle between the vectors much the same  
thing as the dot product? The answer is yes,  they're the same, once you normalize them so.
Is that still on… NICK: It’s correlation what we're  doing here at scale, as well?
JEREMY: Yeah, you can, yeah,  you can think of it that way. NICK: Okay cool…
JEREMY: Now,  
this looks pretty different to how Pytorch looks.   Pytorch has things in rows, right?,  we've got a user, a movie rating,  
user movie rating, right? So, how do we  do the same kind of thing in Pytorch?  
So, let's do the same kind of thing in Excel,  but using the table in the same format that  
Pytorch has it. Okay. So to do that in Excel the  first thing I'm going to do is I'm going to, see,  
okay, this… I've got to look at user number  14 and I want to know what index –like how far  
down this list is 14. Okay, so we'll just match  means find the index. So this is user index one.
And then what I'm going to  do is, I'm going to say the…  
these five numbers is, basically I want to find   row one over here, and in excel that's called  OFFSET. So we're going to offset from here by  
one row, and so you can see here it is 0.19, 0.63,  0.19, 0.63 et cetera, right? So here's the second  
user: 0.25, 0.83 etc. And we can do the same  thing for movies, right? So movie 417 is index 14,  
that's going to be: 0.75, 0.47 et cetera. And so  same thing, right?, but now we're going to offset  
from here, by 14, to get this row which is 0.75,  0.47 et cetera. And so the prediction now is… the  
dot product is called SUMPRODUCT  in excel, this is a SUMPRODUCT   of those two things. So this is exactly the  same as we had before, right?, but when we  
kind of put everything next to each other we  have to, like manually, look up the index.
And so then for each one we can calculate  the error squared prediction minus   rating squared and then we could add those all  up and –if you remember– this is actually the  
same root mean squared error we had before  –we optimized before. 2.81 because we've   got the same numbers as before and  so this is mathematically identical.
So what's this weird word up here:  “embedding”. You've probably heard it before  
What is embedding
and you might have come across the impression  it's some very complex fancy mathematical thing,  
but actually, it turns out, that it is  just looking something up in an array:  
that is what an embedding is. So  we call this an “embedding matrix”,  
and these are our “user embeddings”  and our “movie embeddings”.
So let's take a look at that in Pytorch, and,  you know, at this point if you've heard about  
embeddings before you might be thinking: that  can't be it. And yeah, it's just as complex  
as the rectified linear unit which turned  out to be: replace negatives with zeros.  
Embedding actually means: “look something up in  an array”. So there's a lot of things that we use,  
as deep learning practitioners, to try  to make you as intimidated as possible  
so that you don't wander into our territory  and start winning our Kaggle competitions.  
And unfortunately, once you  discover the simplicity of it,   you might start to think that you can do  it yourself and then it turns out you can.  
So yeah, that's what basically, it turns out  pretty much all of this jargon turns out to be.  
So we're going to try to  learn these latent factors,  
which is exactly what we just did in  Excel, we just learned the latent factors.
All right, so, if we're going to learn things  in Pytorch we're going to need data loaders.  
One thing I did is, there is actually a movies   table as well, with the names of the movies, so  I merged that together with the ratings so that  
then we've now got the user id and the actual name  of the movie –we don't need that, obviously, for   the model, but it's just going to make it a bit  more fun to interpret later–. So this is called:  
ratings. We have something called  CollabDataLoaders –so collaborative filtering data  
loaders– and we can get that from a data frame,  by passing in the data frame, and it expects a  
user column and an item column. So the user column  is what it sounds like: the person that is rating  
this thing, and the item column is the product or  service that they're rating. In our case the user  
column is called “user” –so we don't have to pass  that in– and the item column is called “title”  
–so we do have to pass this in– because by  default the user column should be called “user”   and the item column will be called “item”. Give it  a batch size, and as usual we can call show batch,  
and so, here's our data loaders… a batch  of data loaders –or at least a bit of it–.  
And so now, since we told it the names, we  actually get to see the names which is nice.
All right, so, now we're going to create  the user_factors and movie_factors -ie-  
this one and this one. So the number of  rows of the movie factors will be equal to  
the number of movies and the number of rows of  the user factors will be equal to the number of   users. And the number of columns will be whatever  we want: however many factors we want to create.
John! JOHN: This might be a pertinent  time to jump in with a question:   any comments about choosing the number of factors
How do you choose the number of latent factors
JEREMY: Uhm… not really. We have defaults  that we use –for embeddings– in fastai.  
It's a very obscure formula and people often  ask me for like the mathematical derivation of   where it came from, but what actually happened  is, it's, I wrote down how many factors I think  
is appropriate for different size categories on a  piece of paper in a table –well actually in Excel–  
and then I fitted a function to that and  that's the function. So it's basically a   mathematical function that fits my  intuition about what works well.  
But it seems to work pretty well, I've  seen it’s used in lots of other places now,   lots of papers will be like: “using fastai's rule  of thumb for embedding sizes… here's the formula…”
JOHN: Cool, thank you. JEREMY: It's pretty fast to train  these things so you can try a few.
So we're going to create… so the number  of users is just the length of how many   users there are, number of movies is  the length of how many titles there are;  
so create a matrix of random numbers of  users by five and movies: of movies by five.
And now we need to look up the index of the  movie in our movie latent factor matrix.  
The thing is, when we've learned about  deep learning, we learnt that we do matrix  
multiplications, not lock-something-up  in a matrix –in an array. So in Excel  
we were saying OFFSET, which is to say,  find element number 14 in the table;  
which, that's not a matrix multiply,  how does that work? Well actually it is,  
it actually is for the same  reason that we talked about  
here, which is: we can represent  –find– the element number-one-thing  
–in this list– is actually the same as multiplying  by a one hot encoded matrix. So remember how,  
if we –let's just take off the log for a moment.
Look, this has returned 0.87 –and particularly if  I take the negative off here, if I add this up–  
this is 0.87, which is the result of finding  the index number-one-thing in this list.  
But we didn't do it that way, we did this by  taking the dot product of this (sorry) of this  
and this, but that's actually the same thing,  right? taking the dot product of a one hot encoded  
vector with something is the same as looking  up this index in the vector. So that means that  
this exercise here of looking up the 14th thing  is the same as doing a matrix multiply with a  
one-hot-encoded vector. And we can see that here:  this is how we create a one hot encoded vector  
of length and users in which the third element  is set to one and everything else is zero.
And if we multiply that –so  “at” (@) means, to remember,   matrix multiply in python– so if we  multiply that by our user_factors,  
we get back this answer. And if we  just ask for user_factors number three  
we get back the exact same answer –they're the  same thing. So you can think of “an embedding”  
as being a computational shortcut for multiplying  something by a one-hot-encoded vector.
And so if you think back to what we did with  dummy variables, right, this basically means  
embeddings are like a cool math trick  for speeding up doing matrix multipliers  
with dummy variables. Not just speeding up. We  never even have to create the dummy variables.   We never have to create the one-hot-encoded  vectors. We can just look up in an array.
All right, so we're now ready  to build a collaborative filter…   a collaborative filtering model and  we're going to create one from scratch.  
How to build a collaborative filtering model from scratch
And as we've discussed before,  in Pytorch a model is a class.
And so, we briefly touched on this,  but I've got to touch on it again.   This is how we create a class  in Python. You give it a name,
and then you say how to initialize  it, how to construct it. So in Python,   remember they call these things “dunder  whatever”. This is dunder init, these are magic  
methods that Python will call for you at  certain times. The method called dunder init  
is called when you create an object of this class.  So we could pass it a value, and so now we set the  
attribute called “a equal to that value”, and so  then later on we could call a method called “say”,  
that will say hello to whatever you passed  in here. And this is what it will say. So,  
for example, if you construct an object  of type Example(), passing in Sylvain,  
self.a now equals Sylvain. So if you say use the  dot method, the say method “nice to meet you”,  
x is now “nice to meet you”. So it will  say hello Sylvain, nice to meet you.  
So that's… that's kind of all you need  to know about object-oriented programming   in Pytorch to create a mode. Oh, there's one  more thing we need to know, sorry, which is  
you can put something in  parentheses after your class name,   and that's called the superclass. It's basically  going to give you some stuff for free, give you  
some functionality for free. And if you create  a model in Pytorch, you have to make Module your  
super class. This is actually fastai's version  of Module, but it's nearly the same as Pytorch’s.
So when we create this dot product object, it's  going to call dunder init, and we say well how  
many users are going to be in our model,  and how many movies, and how many factors,  
and so we can now create an  embedding of users by factors   for users, and an embedding of movies  by vectors for movies, and so then
How to understand the `forward` function
Pytorch does something quite magic,  which is that if you create a
dot product object like so, it then… you can  treat it like a function. You can call it  
up and calculate values on  it. And when you do that,   this is really important to know, Pytorch is going  to call a method called “forward” in your class.  
So this is where you put your calculation of  your model. It has to be called “forward”,   and it's going to be passed the object itself,  and the thing you're calculating on. In this case,  
the user and movie for a batch. So this is your  batch of data, each row will be one user and movie  
combination, and the columns will be users  and movies. So we can grab the first column,  
right, so this is every row of the first column,  and look it up in the user factors embedding,  
to get our users embeddings. So that is the same  as doing this. Let's say this is one mini batch.
And then we do exactly the same thing for the  second column, passing it into our movie factors  
to look up the movie embeddings, and  then take the dot product. dim equals one  
because we're summing across the columns for each  row. We're calculating a prediction for each row
so once we've got that we can pass it to a learner, passing  in our data loaders, and our model,  
and our loss function mean squared  error, and we can call fit.
And the way it goes. And this by the way is running on cpu. Now  these are very fast to run. So this is doing  
100 000 rows in 10 seconds, which is a whole  lot faster than our few dozen rows in Excel.
and so you can see the loss going  down. And so we've trained a model. Um…
it's not going to be a great model,   and one of the problems is that, let's  see if we can see this in our Excel one…
Look at this one here. This  prediction is bigger than five.
Adding a bias term
But nothing's bigger than five.  So that seems like a problem.   We're predicting things that are bigger  than the highest possible number.
And in fact these are very much movie  enthusiasts that… nobody gave anything a one.  
Yeah nobody even gave anything a one here. So…
do you remember when we learned about Sigmoid.  The idea of squishing things between zero and one.  
We could do stuff still without a Sigmoid,  but when we added a Sigmoid, it trained better   because the model didn't have to work so  hard to get it, kind of, into the right zone.  
Now if you think about it, if you take something  and put it through a Sigmoid, and then multiply it   by five, now you've got something that's going  to be between zero and five. We used to have  
something which is between zero and one, So we  could do that in fact we could do that in Excel.
I'll leave that as an exercise to the  reader. Let's do it over here in Pytorch.
So if we take the exact same class as before   and this time we call sigmoid_range and  so sigmoid_range is something which will  
take our prediction and then squash it into our  range and by default we'll use a range of zero  
through to 5.5. so it can't be smaller than zero,  can't be bigger than 5.5. Why didn't I use five?  
That's because a sigmoid can never hit one,  right? and a sigmoid times five can never hit five  
but some people do give things... movies a five so  you want to make it a bit bigger than our highest.
Model interpretation
So this one got a loss of 0.8628
86 oh it's not better. Isn't that always the way?   All right, didn't actually  help; doesn't always. So be it.
Let's keep trying to improve it.  Let me show you something I noticed.  
Some of the users, like this one... this  person here just loves movies. They give  
nearly everything a four or five. Their worst  score is a three, all right? This person... oh,  
here's a one. This person's got much more range.  Some things are twos, some ones, some fives.
This person doesn't seem to like movies very  much considering how many they watch. Nothing   gets a five. They've got discerning tastes, I  guess. At the moment we don't have any way in  
our kind of formulation of this model to say this  user tends to give low scores and this user tends  
to give high scores. There's just nothing like  that, right? But that would be very easy to add.
Let's add one more number to  our five factors, just here,  
for each user and now, rather than doing  just the matrix multiply let's add...
Oh it's actually the top one. Let's add this  number to it h19 and so for this one let's  
add i19 to it. Yeah so I've got it wrong.  This one here, so this... this row here...
We're going to add to each rating and then  we're going to do the same thing here.
Each movie's now got an extra number  here that again we're going to  
add a 26. So it's our matrix multiplication  plus, we call it, the bias. The user bias plus  
the movie bias so effectively that's like making  it so we don't have an intercept of zero anymore.
And so if we now train this model...
Data -> Solver -> Solve.
So previously we got to 0.42, okay? and so  we're going to let that go along for a while  
and then let's also go back and  look at the Pytorch version.   So for Pytorch, now we're going to  have a user_bias which is an embedding  
of n_users by 1, right? Remember there was just  one number for each user and movie bias is an  
embedding of n_movies also by 1 and so we can now  look up the user embedding the movie embedding,  
do the dot product and then look up the  user_bias and the movie_bias and add them.
Chuck that through the sigmoid. Let's  train that, see if we beat 0.865.
Wow we're not training very well,  are we? Still not too great. 0.894.   I think Excel normally does  do better though. Let's see.
Okay Excel... oh Excel's done a lot  better. It's gone from 0.42 to 0.35.
Okay so what happened here? Why did it get worse?  Well, look at this the valid loss got better  
and then it started getting worse again.  So we think we might be overfitting,
which you know we have got a lot of parameters in  our embeddings. So how do we avoid overfitting?
So a classic way to avoid overfitting  is to use something called weight decay.
What is weight decay and How does it help
Also known as L2 regularization,  which sounds much more fancy.
What we're going to do is when we can compute the gradients, we're  going to first add to our loss function,  
the sum of the weights squared. This is something  you should go back and add to your titanic model,  
not that it's overfitting, but just to  try it, right? So previously our gradients  
have just been and our loss function has  just been about the difference between our  
predictions and our actuals, right? And so  our gradients were based on the derivative  
of that with respect to the derivative  of that with respect to the coefficients  
but we're saying now “let's add the  sum of the square of the weights  
times some small number”. So what  would make that loss function go down?  
That loss function would go  down if we reduce our weights.
For example if we reduce all  of our weights to zero...  
I should say we reduce the magnitude of our  weights. If we reduce them all to zero, that   part of the loss function will be zero because  the sum of zero squared is zero. Now, problem is  
if our weights are all zero, our model doesn't do  anything, right? So we'd have crappy predictions.  
So it would want to increase the weights so  that's actually predicting something useful.
But if it increases the weights too much then it  starts overfitting. So how is it going to actually  
get the lowest possible value of the  loss function? By finding the right mix.  
Weights not too high, right? But high  enough to be useful at predicting.
If there's some parameter that's not  useful, for example, say we asked for  
five factors and we only need four,   it can just set the weights for the fifth  factor to zero, right? And then problem solved,  
right? It won't be used to predict anything but  it also won't contribute to our weight decay part.
So previously we had something calculating the  loss function. So now we're going to do exactly   the same thing but we're going to square  the parameters, we're going to sum them up,  
and we're going to multiply them by  some small number like 0.01 or 0.001.
And in fact we don't even need to do this   because remember the whole purpose of the loss is  to take its gradient, right? And to print it out.  
The gradient of parameters squared is two times  parameters. It's okay if you don't remember that  
from high school but you can take my word for  it. The gradient of y equals x squared is 2x.  
So actually all we need to do  is take our gradient and add  
the weight decay coefficient 0.01 or whatever  times two times parameters. And given this is just  
number... some number we get to pick, we might as  well pull the 2 into it and just get rid of it.
So when you call fit you can pass in a wd  parameter which does... adds this times the  
parameters to the gradient for you. And  so that's going to ask the model...it's  
going to save the model... “please don't make  the weights any bigger than they have to be”.
And yay! Finally our loss actually improved, okay?  And you can see it getting better and better.
In fastai applications like vision we  try to set this for you appropriately  
and we generally do a reasonably good job. Just  the defaults are normally fine. But in things  
like tabular and collaborative filtering, we don't  really know enough about your data to know what   to use here. So you should just try a few things.  Let’s... try a few multiples of 10. Start at 0.1  
and then divide by 10 a few times, you know, and  just see which one gives you the best result.
What is regularization
So this is called regularization. So  regularization is about making your bottle...  
model no more complex than it has to be, right?  It has a lower capacity and so the higher the  
weights, the more they're moving the model around,  right? So we want to keep the weights down but not  
so far down that they don't make good predictions  and so the value of this if it's higher, will keep  
the weights down more, it will reduce overfitting  but it will also reduce the capacity of your model  
to make good predictions and if it's lower it  increases the capacity of model and increases
overfitting. All right,  
I'm going to take this bit for next time. Before  we wrap up, John, are there any more questions?
JOHN: Yeah there are. The... there's  some from... from back at the start of   the collaborative filtering. So we had a bit of  a conversation a while back about this... the  
size of the embedding vectors and you talked  about your fastai rule of thumb. So there was  
a question if anyone has ever done a kind of  a hyperparameter search, an exploration for…
JEREMY: I mean people often will do  a hyperparameter search, for sure. JOHN: I beg your pardon. JEREMY: People will often do a hyperparameter  search for their model but I haven't seen a...  
I haven't seen any other rules  other than my rule of thumb. JOHN: Right, so not not  productively to your knowledge?
JEREMY: Oh productively for an  individual model that somebody's built. JOHN: And then there's a... there's a question  here from Zakia which I didn't quite wrap my head  
around so Zakia if you want to maybe clarify in  the... in the chat as well but “can recommendation  
systems be built based on average ratings of users  experience rather than collaborative filtering?”
JEREMY: Not really, right? I mean if you've got  lots of metadata, you could, right? So if you've  
got, you know, like lots of information about  demographic data about where the user's from and  
you know what loyalty scheme results they've  had and blah blah blah and then for products   there's metadata about that as well then sure  averages would be fine but if all you've got is  
kind of purchasing history then you really want  the granular data otherwise how could you say  
like... they like this movie, this movie, and this  movie therefore they might also like that movie   or you've got... it's like oh they kind of like  movies. There's just not enough information there.
JOHN: Yeah great. That's about it. thanks okay great alright thanks everybody  see you next time for our last lesson

---

# 8

So welcome to the last lesson of Part One  of Practical Deep Learning for Coders.  
It's been a really fun time doing this  course, and depending on when you're  
watching and listening to this you may want  to check the forums or the fast.ai website  
to see whether we have a Part Two planned,  which is going to be some time towards the  
end of 2022. Or if you're if it's already  past that then maybe there's even a Part Two  
already on the website. So Part Two goes a lot  deeper than Part One technically, in terms of  
getting to the point that you should be able  to read and implement research papers and  
deploy models in a very  kind of real life situation So yeah last lesson we started  
on the Collaborative Filtering notebook and  we were looking at Collaborative Filtering and  
this is where we got to, which is creating your  own embedding module, and this is a very cool…   this is a very cool place to start the  lesson because you're going to learn a  
lot about what's really going on and it's  really important before you dig into this  
to make sure that you're really comfortable with  the 05-liner-model-and-neural-net-from-scratch  
notebook. So if parts of this are not totally  clear, put it aside and redo this notebook  
because what we're looking at from here  are kind of the abstractions that Pytorch  
and fast.ai add on top of… functionality  that we've built ourselves from scratch.  
So if you remember in the neural network  from scratch we built, we initialized a  
number of coefficients, a couple of different  layers, you know, and a bias term and then  
during as the model trained we updated  those coefficients by going through   each layer of them and subtracting out  the gradients by the learning rate…
In… you probably noticed that in Pytorch  we don't have to go to all that trouble,  
and I wanted to show you how Pytorch does this.  Pytorch… we don't have to keep track of what  
our coefficients, or parameters, or weights are -  Pytorch does that for us. And the way it does that  
is it looks inside our module and it tries to  find anything that looks like a neural network  
parameter, or a tensor of neural network  parameters, and it keeps track of them. And  
so here is a class we've created called “T” which  is a subclass of module and I've created one thing  
inside it which is something with the attribute  “a”, so this is “a” in the “T” module and it  
just contains three ones. And so the idea is, you  know, maybe we're creating a module and this is  
we're initializing some parameter that we  want to train. Now, we can find out what  
trainable parameters – or just what parameters  in genera l – Pytorch knows about in our model by  
instantiating our model and then asking for the  parameters – which you then have to turn that  
into a list… or in fastcore we have a thing  called capital “L” which is like a fancy list  
which prints out the number of items in the list  and shows you those items. Now in this case,  
when we create our object of type “T” and ask  for its parameters we get told there are zero  
tensors of parameters and a list with nothing in  it. Now why is that? We actually said we wanted  
to create three… a tensor with three ones  in it. How would we make those parameters?  
Well, the answer is that the way you create… the  way you tell Pytorch what your parameters are  
is: you actually just have to put them inside  a special object called an “nn.Parameter”. This  
Parameters in PyTorch
thing almost doesn't really do anything. In fact  last time I checked it really quite literally had  
almost no code in it... (sometimes these  things change, but let's take a look)...
Yeah okay so it's about a dozen  lines of code or 20 lines of code,  
which does almost nothing. It's got a way of  being copied, it's got a way of printing itself,  
it's got a way of saving itself, and  it's got a way of being initialized.  
So Parameter hardly does anything. The key thing  is though that when Pytorch checks to see which  
parameters should it update when it optimizes,  it just looks for anything that's been wrapped  
in this Parameter class. So, if we do exactly the  same thing as before, which is to set an attribute  
containing a tensor with three ones… in it  but this case we wrap it in a Parameter.
We now get told: okay,  there's one parameter tensor   in this model and it contains a tensor with  three ones. And you can see it also actually,  
by default, assumes that we're going to want  –require– a gradient. It's assuming that   anything that's a parameter is something  that you want to calculate gradients for.  
Now, most of the time we don't have to  do this because Pytorch provides lots of  
convenient things for us such as what you've  seen before “nn.Linear”, which is something  
that also (contains) creates a tensor. So this  would (contain) create a tensor of one by three  
without a bias term in it. This has not  been wrapped in a “nn.Parameter” – but  
that's okay. Pytorch knows that anything  which is basically a layer in a neural net  
is going to be a parameter, so it  automatically considers this a parameter.
So here's exactly the same thing again,  I construct my object of type “T”,   I'll check for its parameters and I can  see there's (three of…) one tensor of  
parameters and there's our three things. And  you'll notice that it's also automatically,  
randomly, initialized them, which again is  generally what we want. So Pytorch does go to some  
effort to, yeah, try to make things easy for you.
So the, this attribute “a” is a… is a  
linear layer and it's got a bunch of things  in it. One of the things in it is the weights,  
and that's where you'll actually find the  parameters, that is, of type “Parameter”.   So a linear layer is something that  contains attributes of type “Parameter”.
Embedding from scratch
Okay so what we want to do is, we  want to create something that works  
just like this did: which is  something that creates a matrix  
which will be trained as we train the model…
Okay so, an embedding is something which, yeah,  it's going to create a matrix of this by this,  
and it will be a parameter and it's something  that, yeah, we need to be able to index into,  
as we did here. And so how, yeah, what is… what is  happening behind the scenes, you know, in Pytorch?  
It's nice to create these things ourselves from  scratch because it means we really understand it.
And so, let's create that exact same  module that we did last time but this time  
we're going to use a function I've  created called “create_params()”.   You pass in a size, so… such as…  in this case – n uses by n factors.  
And it's going to call “torch.zeros()”  to create a tensor of zeros  
of the size that you request,  and then it's going to do a  
normal random distribution… so a gaussian  distribution… of mean zero standard deviation 0.01  
to randomly initialize those, and it'll  put the whole thing into an “nn.Parameter”.  
So that… so this here is going to create  an attribute called “user_factors” which  
will be a parameter containing some tensor of  normally distributed random numbers of this size.
Excuse me. And because it's a parameter that's going  to be stored inside… that's going to be  
available as in parameters in the “Module”...
(Oh why am I sneezing) So “user_bias” will be a vector of parameters,  “user_factors” will be a matrix of parameters,  
“movie_factors” will be a matrix “n_movies” by  “n_factors”, “movie_bias” will be a vector of  
“n_movies” and this is the same as before.  So now in the forward() we can do exactly  
what we did before. The thing is, when  you put a tensor inside a “Parameter”  
it has all the exact same features that a tensor  has. So, for example, we can index into it.
So this whole thing is  identical to what we had before   and so that's actually, believe it or  not, all that's required to replicate  
Pytorch's embedding layer from scratch.  So let's run those and see if it works.  
And there it is: it's training. So we'll be able  to have a look when this is done, at for example…  
((let's have a look)) model.move_bias.
And here it is, right?,   it's a “Parameter” containing a bunch  of numbers that have been trained.
And as we'd expect it's got 1665 things  in, because that's how many movies we have.
So a question from Jona Raphael was:  “does ‘torch.zeros’ not produce all zeros?
Yes “torch.zeros” does produce all zeros. But  remember, a method that ends in underscore changes  
in-place the tensor it's being applied  to. And so, if you look up “pytorch  
normal_” you'll see it fills itself  
with elements sampled from the normal  distribution. So this is actually modifying  
this tensor in place, and so that's why we  end up with something which isn't just zeros.
Now this is a bit I find really fun, is,  we train this model, but what did it do?
Embedding interpretation
How is it going about predicting who's going  to like what movie? What, well, one of the  
things that's happened is we've created this  “movie_bias” parameter which has been optimized  
and what we could do is we  could find which Movie-IDs  
have the highest numbers here; and the  lowest numbers, (so I think this is going   to start lowest) and then we can print out…  we can look inside our data loaders and grab  
the names of those movies for  each of those five lowest numbers.
And what's happened here? Well, we  can see broadly speaking that it has   printed out some pretty crappy movies.  And, why is that? Well, that's because  
when it does that matrix product that we saw in  the excel spreadsheet last week, it's trying to  
figure out who's going to like what movie based  on previous movies people have enjoyed or not,  
and then it adds “movie_bias”,  which can be positive or negative,   that's a different number for each movie.  So in order to do a good job of predicting  
whether you're going to like a movie or not, it  has to know which movies are crap, and so the   crap movies are going to end up with a very low  “movie_bias” parameter, and so we can actually  
find out which movies, do people… not only  which movies do people really not like,  
but which movies do people like less than one  would expect given the kind of movie that it is.  
So “Lawnmower Man 2”, for example, not only  apparently is it a crappy movie but, based on the  
kind of movie it is (you know, it's kind of like  a high-tech pop kind of sci-fi movie…) …people  
who like those kinds of movies still don't like  “Lawnmower Man 2”, so that's what this is meaning.
So it's kind of nice that we can, like, use  a model not just to predict things but to   understand things about the data. So if we sort  by descending, it'll give us the exact opposite.  
So here are movies that people enjoy   even when they don't normally enjoy that kind  of movie. So for example “L.A. Confidential”,  
classic kind of film noir detective movie with the  aussie Guy Pierce, even if you don't really like  
film noir detective movies, you might like  this one. You know, “Silence of the Lambs”,  
classic kind of… I guess you'd say like, horror  kind of… not horror is it, a suspense movie… even  
people who don't normally like kind-of serial  killer suspense movies tend to like this one.
Now, the other thing we can do is  not just look at what's happening in   the bias… oh and by the way, we could do the same  thing with users and find out, like, which user  
just loves movies, even the crappy  ones, you know. Dislikes all movies  
and vice versa. But what about the other  thing – we didn't just have bias – we also had  
movie factors. Which has got the number of movies  as one axis and the number of factors as the other  
and we passed in 50 – what's in that huge matrix?  Well, pretty hard to visualize such a huge matrix  
and we're not going to talk about the  details, but you can do something called PCA,   which stands for Principal Component  Analysis, and that basically tries to  
compress those 50 columns down into three columns…  and then we can draw a chart of the top two.
And so this is PCA component number one  and this is PCA component number two,  
and here's a bunch of movies and this is a  compressed view of these latent factors that  
it created. And you can see that they  obviously have some kind of meaning,   right? So over here towards the right, we've got  kind of, you know, very pop mainstream kind of  
movies. And over here on the left, we've got more  of the kind of critically acclaimed gritty kind  
of movies. And then towards the top we've got very  kind of action-oriented and sci-fi movies and then  
down towards the bottom we've got very dialogue  driven movies. So remember, we didn't program in  
any of these things and we don't have any data  at all about what movie is what kind of movie  
but thanks to the magic of SGD, we  just told it to please try and optimize  
these parameters and the way it was able  to predict who would like what movie  
was it had to figure out what kinds of movies  are there or what kind of taste is there  
for each movie. So I think  that's pretty interesting.  
So this is called “visualizing embeddings”  and then this is “visualizing the bias”.
Collab filtering in fastai
We…
We obviously would rather not do everything  both by hand, like this, or even like this,  
and fast.ai provides an application for  collaborative learner [“collab_learner”].  
And so we can create one, and this is going  to look much the same as what we just had,   we're going to say how many latent  factors we want and what the “y_range” is:  
to do the sigmoid and the multiply; and  then we can do “fit”, and away it goes…
So let's see how it does. All right, so, it's done a bit better than our  
manual one. Let's take a look at the model  it created. The model looks very similar  
to what we created in terms of the parameters.  You can see here, these are the two embeddings   and these are the two biases, and we can do  exactly the same thing… we can look in that  
model and we can find the – you'll see  it's not called movies it's “i” for items,  
it uses an items… this is the item_bias. So we  can look at the item_bias, grab the weights,  
sort, and we get a very similar result. In  this case it's very… even more confident  
that “L.A. Confidential” is a movie that you  should probably try watching even if you don't   like those kind of movies. And “Titanic” is right  up there as well. Even if you don't really like  
romancy kind of movies, you might like this one. Even if you don't like classic  detective, you might like this one  
[Pointing “Rear Window”] You know, we can have a look at the source  code for “collab_learner” and we can see that…
Let's see, “use_nn” is false by default, so  where our model is going to be of this type…  
“EmbeddingDotBias”... so  we can take a look at that…  
Here it is, and look!, this does look very  similar, okay? It's creating an embedding,  
using the size we requested for each of  users-by-factors and items-by-factors and  
users and items. And then it's wrapping each  thing from the embedding in the “forward”,  
and it's doing the multiply, and it's  adding it up, and it's doing the sigmoid.  
So yeah, it looks exactly the same,  isn't that neat?. So you can see that  
what's actually happening in real models is not,  yeah, it's not… it's not that weird or magic.
So Kurian is asking: “is PCA useful in  any other areas?” And the answer is:  
absolutely! And what I suggest you do  
–if you're interested– is check out our  (~contra) “Computational Linear Algebra” course.
It's five years old now but it... I mean… this  is stuff which hasn't changed for decades really  
and this will teach you all about things  like PCA and stuff like that. It's not nearly  
as directly practical as “Practical Deep  Learning for Coders” but it's definitely,   like, very interesting and it's the kind  of thing which, if you want to go deeper,  
you know, it can become pretty  useful later along your path.
Embedding distance
Okay, so here's something  else interesting we can do:   let's grab the “movie_factors”. So that's in our  model, it's the item weights [“i_weights”] and  
it's the “weight” attribute that Pytorch  creates. Okay, and now we can convert  
the movie “Silence of the Lambs” into its  “class_id” and we can do that with object-to-id  
“o2i”, for the titles. And so that's the movie  index of “Silence of the Lambs”. And what we can  
do now is we can look through all of the movies  in our latent factors and calculate how far apart  
(~the) each vector is, each embedding vector is  from this one. And this “CosineSimilarity” is  
very similar to basically the “Euclidean  Distance”, you know, the kind of the  
Root Sum Squared of the… of the  differences, but it normalizes it.  
So it's basically the angle between the vectors.  So this is going to calculate how similar each   movie is to the “Silence of the Lambs” on, based  on these latent factors. And so then we can find  
which ID is the closest. Yeah, so  based on this embedding distance,  
the closest is “Dial M for Murder”…  which makes a lot of sense.
I'm not going to discuss it today but  in the book there's also some discussion   about what's called the Bootstrapping  Problem, which is the question of, like,  
if you've got a new company, or a new  product, how would you get started with  
making recommendations given that you don't  have any previous history with which to make   recommendations and that's a very interesting  problem that you can read about in the book.
Collab filtering with DL
Now… that's one way to do Collaborative  Filtering, which is where we create that…  
do that matrix completion exercise using all those  dot products. There's a different way however,  
which is we can use Deep Learning. And to do  it with Deep Learning, what we could do is, we  
can… we could basically create our user and item  embeddings as per usual and then we could create  
a sequential model. So a sequential model is just  layers of a Deep Learning Neural Network in order.
And, what we could do is we  could just concatenate… so   in “forward” we could just concatenate  the user and item embeddings together,  
and then do a reLU. So this is… this is basically  a single Hidden Layer Neural Network, and then a  
linear layer at the end to create a single output.  So this is the very, you know, world's most simple  
neural net exactly the same  as the style that we created  
back here in our “...neural net from  scratch.” This is exactly the same,  
but we're using Pytorch's  functionality to do it more easily.
So, in the “forward” here we're gonna, in the  same… exactly the same way as we have before,   we'll look up the user embeddings and  we'll look up the item embeddings and  
then this is new, this is where we  concatenate those two things together   and put it through our Neural Network  and then finally do our sigmoid.
Now, one thing different this  time is that we're going to  
ask fast.ai to figure out how big our embeddings  should be, and so fast.ai has something called  
get embedding sizes [“get_emb_sz”]. And it  just uses a rule of thumb that says that: for   944 users we recommend 74 factor embeddings and  for 1,665 movies (or is it the other way around,  
I can't remember) we recommend 102 factors for  your embeddings – so that's what those sizes are.  
So now we can create that model and we can pop  it into a learner and fit in the usual way.
And so, rather than doing all that from scratch,   what you can do is you can do exactly  the same thing that we've done before:  
which is to call collaborative learner  [“collab_learner”], but you can pass in the   parameter use neural network [“use_nn”] equals  True, and you can then say how big do you want  
each layer. So this is going to create a two  hidden layer Deep Learning Neural Net: the first   will have (~1500) [100] and the second will have  50. And then you can say “fit” and away it goes.
Okay so here is our… oh we got 0.87… so  these are doing less well than our dot  
product version which is not too surprising  because kind of the dot product version is   really trying to take advantage of our  understanding of the problem domain.  
In practice, nowadays, a lot of companies kind of  combine…they kind of create a combined model that  
have a has a Dot Product component and also  has a Neural Net component. The Neural Net  
component is particularly helpful if you've  got metadata, for example information about  
your users, like: when did they sign up; how old  are they; what sex are they; you know, where are  
they from. And then those are all things that  you could concatenate in, with your embeddings,  
and ditto with metadata about the movie: how  old is it; what genre is it; and so forth.
All right, so, we've got a question  from Jona which I think is interesting.   And the question is: “is there an  issue where the bias components are  
overwhelmingly determined by  the non-experts in a genre?”
In general, actually there's a more general  issue which is, in Collaborative Filtering  
Recommendation Systems, very often a small  number of users or a small number of movies  
overwhelm everybody else and the classic one is  Anime. A relatively small number of people watch  
Anime and those group of people watch a lot of  Anime. So in movie recommendations, like, there's  
a classic problem which is every time people  try to make a list of well-loved movies all the  
top ones seem to be anime. And so you can imagine  what's happening in the matrix completion exercise  
is that there are... yeah  some… some users that just,  
you know, really watch this one genre of  movie and they watch an awful lot of them.  
So in general you actually do have to  be pretty careful about the, you know,   these subtlety kind-of issues. And yeah I won't  go into details about how to deal with them but  
they generally involve taking various kinds  of ratios, or normalizing things or so forth.
Embeddings for NLP
All right, so, that's Collaborative Filtering,  and, I wanted to show you something interesting  
then about embeddings, which is that embeddings  are not just for Collaborative Filtering. And  
in fact, if you've heard about embeddings  before you've probably heard about them in   the context of Natural Language Processing.  So you might have been wondering, back when  
we did the Hugging Face transformers stuff, how  did we go about, you know, using text as inputs  
to models. And we talked about how you can  turn words into integers… we make a list…
So here's… here's the movie (sorry...)  here's the the poem “I am Sam”:  I am Daniel / I am Sam / Sam I am /  That Sam-I-am, et cetera, et cetera.
We can find a list of all  the unique words in that poem   and make this list here. And then  we can give each of those words  
a unique id, just arbitrarily, well actually in  this case it's alphabetical order, but it doesn't  
have to be. And so we kind of talked about that  and that's what we do with categories in general,  
but how do we turn those into like, you know,  lists of random numbers. And you might not be  
surprised to hear, what we do is: we create  an embedding matrix. So here's an embedding  
matrix containing four latent factors for each  word in the vocab. So here's each word in the  
vocab and here's the embedding matrix. So if we  then want to present this poem to a Neural Net…  
then what we do is we list out  our poem – I do not like that,  
Sam-I-am / Do you like Green eggs and ham, etc.  Then for each word we look it up… so in Excel,  
for example, we use “MATCH”, so that will find  this word over here and find it is word id 8  
and then we will find the 8th word and  the first embedding. And so that gives us…
(that's not right, eight?)...
Oh no, that is right, sorry. Here it  is, it's just weird column widths.   So it's going to be 0.22, then 0.1, 0.01 and  here it is 0.22, 0.1, 0.01 etc. So this is the  
embedding matrix we end up with for this  poem. And so if you wanted to train,  
or use a trained neural network on this poem, you  basically turn it into this matrix of numbers.  
And so, this is what an embedding matrix looks  like in an NLP model and it works exactly the same  
way, as you can see. And then you can do exactly  the same things in terms of interpretation of an  
NLP model by looking at both the bias factors and  the latent factors in a word embedding matrix.
So hopefully you're getting the idea here that…  
our, you know, our different models, you know,  the inputs to them… they're based on a relatively  
small number of, kind of, basic principles.  And these principles are generally things like:  
look up something in array. And then we  know, inside the model, we're basically  
multiplying things together, adding them  up and replacing the negatives with zeros.   So hopefully you're getting the idea that  what's going on inside a Neural Network is  
generally not that complicated, but  it happens very quickly and at scale.
Now, it's not just Collaborative Filtering  and NLP but also Tabular Analysis.  
Embeddings for tabular
So, in Chapter 9 of the book we've talked  about how Random Forests can be used for this,  
which was for… this is for the thing where we're  predicting the auction sale price of industrial  
heavy equipment like bulldozers. Instead of  using a Random Forest, we can use a Neural Net.
Now, in this data set, there are some  continuous columns and there are some  
categorical columns. Now, I'm not going to  go into the details too much, but in short,  
we can separate out the continuous columns  and categorical (~problem) columns using   “cont_cat_split” and that will automatically  find which is which based on their data types.
And so in this case it looks like…   okay so continuous columns, the elapsed sale  date, so I think it's the number of seconds,  
or years, or something since the start of  the data set, is a continuous variable.
And then here are the “cat”,  the categorical variables.   So for example there are six different  product sizes and two coupler systems,  
5059 model descriptions, 6 enclosures,  17 tire sizes and so forth.
So, we can use fast.ai, basically, to  say: okay, we'll take that data frame  
and pass in the categorical and continuous  variables, and create some random splits,  
and what's the dependent variable, and  we can create data loaders from that.  
And from that we can create a “tabular_learner”;  and basically what that's going to do is… it's  
going to create a pretty regular, multi-layer  neural network, not that different to  
this one, that we created by hand.
And each of the categorical variables, it's  going to create an embedding for it. And so I  
can actually show you this, right? So we're going  to use “tabular_learner” to create the “learner”  
and so “tabular_learner” is one, two, three, four,  five, six, seven, eight, nine lines of code. And  
basically the main thing it does is create a  “TabularModel”. And so then “TabularModel”,  
you're not going to understand all of it,  but you might be surprised at how much.   So a “TabularModel” is a module, we're going to be  passing in how big is each embedding going to be,  
and “tabular_learner”... what's  that passing in? It's gonna call  
get embedding sizes [“get_emb_sz”] just like  we did manually before –automatically. So  
that's how it gets its embedding sizes and  then it's going to create an “Embedding”  
for each of those embedding sizes [“emb_szs”],  from number of inputs, to number of factors.
“Dropout” we're going to come back to later.  “BatchNorm” we won’t do until part two.
So then it's going to create a layer, for each of  the layers we want, which is going to contain a  
linear layer, followed by batch norm, followed  by drop out [“LinBnDrop”]. It's going to add the  
“SigmoidRange” we've talked about at the very end.  And so the “forward()”, this is the entire thing!  
If there's some embeddings, it'll go through  and get each of the embeddings using the same   indexing approach we've used before.  It'll concatenate them all together,  
and then it'll run it through the layers  of the neural net, which are these.
So yeah, we don't know all of those details  yet, but we know quite a few of them.  
So that's encouraging, hopefully.  
And once we've got that, we can do  the standard “lr_find” and “fit”.
Now, this exact dataset was used in a Kaggle  competition… this dataset was in a Kaggle  
competition… and the third place getter  published a paper about their technique  
and it's basically the exact, almost the exact  one, I'm showing you here. So it wasn't this,  
sorry, it wasn't this data set, it was a  dataset, it was a different one, it was about  
predicting the amount of  sales in different stores…
But they, they used this basic kind of technique.   And one of the interesting things  is that they used a lot less manual  
feature engineering than the other high placed  entries. Like they had a much simpler approach,  
and one of the interesting things they  published a paper about their approach.  
So they published a paper about their approach.  
So this is the team, from this company,   and they basically describe here exactly what  I just showed you – these different embedding  
layers being concatenated together and then going  through a couple of layers of a neural network,  
and it's showing here… it points out in the paper  exactly what we learned in the last lesson, which   is, embedding layers are exactly equivalent to  linear layers on top of a one-hot encoded input.  
And, yeah, they found that their technique worked  really well. One of the interesting things they  
also showed is that you can take, you can create  your Neural Net, get your trained embeddings,  
and then you can put those embeddings into  a Random Forest or Gradient Booster Tree  
and your Mean Average Percent Error will  dramatically improve. So you can actually combine  
Random Forests and embeddings or Gradient Boosted  Trees and embeddings, which is really interesting.
Now, what I really wanted to show you though is,  what they then did… so, as I said this was a thing   about the predicted amount that different products  would sell for at different shops around Germany.  
And what they did was they had a… so, one of  their embedding matrices was embeddings by region,  
and then they did… I think this is a PCA  (Principal Component Analysis) of the embeddings  
for their German regions and when they create a  chart of them, you can see that the locations that  
close together in the embedding matrix are the  same locations that are close together in Germany.  
So you can see here's the blue ones and here's  the blue ones. And again, it's important to   recognize that the data that they used had no  information about the location of these places,  
the fact that they are close together  geographically is something that was  
figured out as being something that  actually helped it to predict sales.
And so, in fact, they then did a plot showing:  each of these dots is a shop (a store)  
and it's showing: for each pair  of stores, how far away is it  
in real life (in metric space) and then how  far away is it in embedding space. And there's  
this very strong correlation, right? So, it's,  you know, it's kind of reconstructed somehow  
(~this kind of) the kind of the geography of  Germany, by figuring out how people shop. And  
similar for days of the week, so  there was no information, really,   about days of the week but when they put it  on the embedding matrix the days of the week:  
Monday, Tuesday, Wednesday – close to each  other. Thursday, Friday – close to each other.   As you can see, Saturday and Sunday – close to  each other. And ditto for months of the year:  
January, February, March, April, May, June. So  yeah, really interesting, cool stuff, I think,  
what's actually going on inside a Neural Network.
All right, let's take a 10 minute break  and I will see you back here at 7:10.
Convolutions
All right folks, this is   something I think is really fun, which  is: we're going to… we've looked at  
what goes into the start of a model “the input”.  We've learned about how they can be categories,  
or embeddings, and embeddings are basically  kind of one-hot-encoded (~category)   categories with a little compute trick,  or they can just be continuous numbers.  
We've learned about what comes (~the other) at  the other side, which is a bunch of activations,   so just a bunch of tensors of numbers which we  can use things like softmax to constrain them to  
add up to one and so forth. And we've looked  at what can go in the middle, which is the…  
matrix multiplies, sandwiched together  with, you know, as rectified linear units.  
And I mentioned that there are other  things that can go in the middle   as well but we haven't really talked  about what those other things are.
So I thought we might look at one of  the most important and interesting   version of things that can go in the middle,  
but what you'll see is it turns out it's actually  just another kind of matrix multiplication.  
Which might not be obvious at first but I'll  explain. We're going to look at something   called a convolution – and convolutions are at  the heart of a convolutional neural network.  
So, the first thing to realize is a  convolutional neural network is very,   very, very similar to the neural networks  we've seen so far – it's got inputs, it's got  
things that are a lot like, or actually  are a form of, matrix multiplication   sandwiched with activation functions  which can be rectified linear.  
But there's a particular thing which makes them  very useful for computer vision. And I'm going  
to show you using this excel spreadsheet  that's in our repo called “conv-example”.  
And we're going to look at it using an image from  MNIST. So MNIST is kind of the world's most famous  
computer vision dataset (I think) because it was  like the first one, really, which really showed  
image recognition being… being cracked.  It's pretty small by today's standards,   it's a dataset of handwritten digits, each one is  28 by 28 pixels. But it, yeah, you know, back in  
the mid 90s Yann LeCun showed, you know, really  practically useful performance on this dataset  
and as a result ended up with convnets being used  in the american banking system for reading checks.
So here's an example of one of those digits,  this is a seven that somebody drew – it's one   of those ones with a stroke through it – and this  is what it looks like. This is… this is the image.  
And so I got it from MN… this is just one of  the images from MNIST which I put into Excel.
And what you see in the… in the next column  
is a version of the image where the horizontal  lines are being recognized and another one where  
the vertical lines are being recognized. And  if you think back to that Zeiler and Fergus   paper that talked about what the layers of a  neural net does, this is absolutely an example  
of something that we know that the first layer  of a neural network tends to learn how to do.  
Now, how did I do this? I did this using something  called a convolution and so, what we're going to  
do now is, we're going to zoom in to this excel  notebook. We're going to keep zooming in. We're  
going to keep zooming in. So take a look.. keep  an eye on this… on this image, and you'll see that   once we zoom in enough, it's actually just made of  numbers which, as we discussed in the very first…  
in the very first lesson, we saw  how images are made of numbers.
So here they are, right? Here are  the numbers, between zero and one.   And what I just did is, I just used a  little trick, I used Microsoft Excel's  
conditional formatting to basically make things:  the higher numbers more red. So that's how I turn  
this Excel sheet and I've just rounded it off  to the nearest decimal, but it's actually,  
they're actually bigger than that. So  yeah, so here is the image as numbers.
And so let me show you how we went about creating  this top edge detector. What we did was we created  
this formula… (don't worry about  the MAX…) let's focus on this...  
what it's doing is… have a look at the colored  in areas… it's taking each of these cells  
and multiplying them by each of  these cells, and then adding them up.  
And then we do the rectified linear part  which is, if that ends up less than zero,  
then make it zero. So this is a,  this is like a rectified linear unit  
but it's not doing the normal matrix  product, it's doing the equivalent  
of a dot product but just on these nine  cells, and with just these nine weights.
So you might not be surprised to hear  that if I move now one to the right,  
then now it's using the next nine  cells, all right? So if I move, like,   to the right quite a bit and down quite a  bit to here, it's using these nine cells.  
So it's still doing a dot product, right? …which  as we know is a form of matrix multiplication,  
but it's doing it in this way where it's kind  of taking advantage of the… of the geometry   of this situation that the things that are  close to each other are being multiplied  
by this consistent group of the same nine  weights each time, because there's actually  
28 by 28 numbers here, right? Which I think is  768... 28 times 28. That’s close enough, 784,  
but (~we don't want, we're not) we don't have  784 parameters, we only have nine parameters.   And so, this is called a convolution. So  a convolution is where you basically slide  
this, kind of, little 3 by 3 matrix across  a bigger matrix and at each location  
you do a dot product of the corresponding elements  of that 3 by 3 with the corresponding elements of  
this 3 by 3 matrix of coefficients. Now, why does  that create something that finds, as you see, top  
edges. Well, it's because of the particular way  I constructed this 3 by 3 matrix. What I said was  
that all of the rows, just above, (so  these ones) are going to get a one,  
and all of the ones just below are going to  get a minus one, and all of the ones in the   middle are going to get a zero. So let's think  about what happens somewhere like here, right?  
That is (let's try to find the right one…  here it is…) …so here we're going to get  
(1 x 1) + (1 x 1) + (1 x 1) - (1 x 1) -  (1 x 1) - (1 x 1). We're going to get 0.
But what about up here. Here we're going  to get (1 x 1) + (1 x 1) + (1 + 1).  
These do nothing because they're ( x 0),  minus (1 x 0). So we're going to get 3.  
So we're only going to get 3 (the highest  possible number) in the situation where  
these are all as black as possible,  or in this case as red as possible,   and these are all white. And so that's  only going to happen at a horizontal edge.
So the one underneath it does exactly the same  thing, exactly the same formulas (oopsie daisy.) 
The one underneath are exactly the same formulas,  a 3 by 3 sliding thing here, but this time we've  
got a different mat… different little mini  matrix of coefficients which is all ones going  
down and all minus ones going down. And so, for  exactly the same reason, this will only be three  
in situations where they're all  one here and they're all zero here.
So, you can think of a convolution   as being a sliding window of little mini dot  products of these little 3 by 3 matrices.  
And they don't have to be 3 by 3, right?  You could have, we could just have easily   done 5 by 5 and then we'd have a 5 by 5 matrix of  coefficients –or whatever, whatever size you like.  
So the size of this is called its kernel size.  This is a 3 by 3 kernel for this convolution.
So then, because this is deep learning,   we just repeat the… we just repeat  these steps again, and again and again.  
So this is… this layer I'm calling “Conv1”  – it's the first convolutional layer.   So “Conv2” is going to be a little bit different  because on “Conv1” we only had a single channel  
input: it's just black and white or, you know,  yeah, black and white, grayscale, one channel.
But now we've got two channels. We've got the   (let's make it a little smaller so we can see  better). We've got the horizontal edges channel  
and the vertical edges channel. And would have  a similar thing in the first layer of its color,  
we'd have a red channel, a green  channel and blue channel. So now our  
filter (this is called the filter, this little  mini matrix is called the filter), our filter…  
our filter now contains a 3 x 3 x depth 2  (or if you want to think of another way)  
two 3 x 3 kernels, or one 3 x 3 x 2 kernel.  And we basically do exactly the same thing,  
which is we're going to multiply each of these, by  each of these, and sum them up. But then we do it  
for the second bit as well, we multiply each  of these, by each of these, and sum them up.  
And so that gives us… and I think I just picked  some random numbers here, right?. So this is going  
to now be something which can combine… oh sorry,  the set… the second one, the second set… so it's,  
sorry… each of the red ones, by each of the blue  ones (that's here) plus each of the green ones,  
times each of the mauve ones (that's here).  So this first filter is being applied to the  
horizontal edge detector and the second filter  is being applied to the vertical edge detector  
and as a result, we can end up with something  that combines features of the two things.
And so then we can have a second channel  over here, which is just a different bunch of  
convolutions for each of the two channels, this  one times this one, again, you can see the colors.
Optimizing convolutions
So, what we could do is if, you know,  once we kind of get to the end we'll  
end up (as I'll show you how in a  moment) we'll end up with a single  
set of ten activations, one per num-digit we're  recognizing – zero to nine. Or, in this case I  
think we could just create one, you know, maybe  we're just trying to recognize nothing but the   number seven, or not the number seven. So we  could just have one activation. And then we would  
back propagate through this using SGD in the usual  way. And that is going to end up optimizing these  
numbers. So in this case I manually put in the  numbers I knew would create edge detectors.  
In real life you start with random numbers and  then you use SGD to optimize these parameters.
Pooling
Okay, so there's a few things we can do next  and I'm gonna, I'm gonna show you the way that  
was more common a few years ago and then  I'll explain some changes that have been   made more recently. What happened a few years  ago was we would then take these activations,  
which as you can see these activations  now are kind of in a grid pattern,  
and we would do something called Max Pooling. And  Max Pooling is kind of like a convolution (it's  
a sliding window) but this time as the sliding  window goes across (so here, we're up to here)  
we don't do a dot product over a filter, but  instead we just take a maximum (see here),  
just this is the maximum of these four  numbers, and if we go across a little bit  
this is the maximum of these four numbers, go  across a bit, go across a bit, and so forth  
(oh that goes off the edge). And you can see what  happens when this is called a 2 by 2 Max Pooling.
So, you can see what happens, with the  2 by 2 max pooling we end up losing  
half of our activations on each dimension, so  we're going to end up with only one quarter  
of the number of activations we used to have. And  that's actually a good thing because if we keep on  
doing convolution, max pool, convolution, max  pool, we're going to get fewer and fewer and  
fewer activations, until eventually, we'll  just have one left, which is what we want.  
That's effectively what we used to do, but, the  other thing I mentioned is we didn't normally keep  
going until there's only one left. What we used  to then do is we'd basically say: okay, at some  
point, we're going to take all of the activations  that are left and we're going to basically  
just do a dot product of those with a bunch of  coefficients, not as a convolution but just as a  
normal linear layer (and this is called the Dense  Layer) and then we would add them all up. So we  
basically end up with a final big dot product  of all of the max pooled activations by all of  
the weights, and would do that for each channel  and so that would give us our final activation.  
And as I say here, MNIST would actually have  10 activations so you'd have a separate set   of weights for each of the digits you're  predicting and then softmax after that.
Okay, nowadays we do things very slightly  differently, nowadays we normally don't   have Max Pool layers but instead, what we  normally do is, when we do our sliding window  
(like this one here) we don't normally…  let's go back to see… so when I go  
one to the right… so currently we're starting  in cell column “G”, if I go one to the right  
the next one is column “H”, and if I go one  to the right the next one starts in column   “I”. So you can see it's sliding the window  over every 3 by 3. Nowadays what we tend to  
do instead is we generally skip one, so we  would normally only look at every second.  
So we would after doing column “I”, we would skip  columns “J” and would go straight to column “K”  
and that's called a Stride 2 Convolution. We do  that both across the rows and down the columns.   And what that means is every time we do a  convolution, we reduce our effective kind  
of feature size (grid size) by two on each  axis, so it reduces it by four in total.  
So that's basically, instead of doing Max Pooling.  And then the other thing that we do differently is  
nowadays we don't normally have a single dense  layer at the end (a single matrix multiply at  
the end) but instead what we do… we generally  keep doing stride 2 convolutions so each one is  
going to reduce the grid size by 2 x 2, we keep  going down until we've got about a 7 by 7 grid  
and then we do a single pooling at the end. And  we don't normally do max pool nowadays – instead  
we do an average pool. So we average the  activations of each one of the 7 by 7 features.
This is actually quite important to know because  if you think about what that means, it means that  
something like an ImageNet style image detector  is going to end up with a 7 by 7 grid – let's  
say it's trying to say, is this a bear? –  and in each of the parts of the 7 by 7 grid   it's basically saying: is there a bear in  this part of the photo?, is there a bear  
in this part of the photo? is there a bear in  this part of the photo? And then it takes the   average of those 49 (7 x 7) predictions to  decide whether there's a bear in the photo.  
That works very well if it's basically  a photo of a bear, right, because most,  
you know, if it's… if the bear is big and takes up  most of the frame then most of those 7 by 7 bits  
are bits of a bear. On the other hand,  if it's a teeny tiny bear in the corner,  
then potentially only one of those 49 squares  has a bear in it, and even worse if it's like  
a picture of lots and lots of different things,  only one of which is a bear, it could end up not   being a great bear detector. And so this is where,  like, the details of how we construct our model  
turn out to be important. And so, if you're trying  to find, like, just one part of a photo that  
has a small bear in it, you might decide to  use (~average pool… sorry) maximum pooling  
instead of average pooling because max-pooling  will just say: I think this is a picture of a  
bear if any one of those 49 bits of my grid  has something that looks like a bear in it.
So these are, you know, these are potentially  important details which often get hand waved over.
Although, you know, again like, the key thing here  is that… this is happening right at the very end,  
right, that max pool or that average pool, and  actually fast.ai handles this for you – we do a   special thing which we, kind of, independently  invented (I think we did it first) which is,  
we do both max pool and average pool and we  concatenate them together. We call that Concat  
Pooling and that has since been reinvented in  at least one paper. And so that means that you  
don't have to think too much about it because  we're going to try both for you, basically.
Convolutions as matrix products
So I mentioned that this is actually really  just matrix multiplication and to show you that,  
I'm going to show you some  images created by a guy called   Matt Kleinsmith who did this… actually,  I think this was in our very first ever  
course (might have been the Part 2, first Part  2 course) …and he basically pointed out that  
in a certain way of thinking about it, it turns  out that convolution is the same thing as a matrix  
multiplier. So I want to show you how he shows  this. He basically says, okay, let's take this  
3 by 3 image and a 2 by 2 kernel containing  the coefficients alpha, beta, gamma, delta.
And so, in this… as we slide the  window over, each of the colors…  
each of the colors are multiplied  together: red by red, plus green by green,   plus (what is that?) orange  by orange, plus blue blue,  
gives you this. And so to put it another way  algebraically P = alpha x A + beta x B… et cetera.
And so then, as we slide to this part, we're  multiplying again: red by red, green by green and  
so forth, so we can say Q = alpha x B + beta x C  etc. And so this is how we calculate a convolution  
using the approach we just  described, as a sliding window.
But here's another way of thinking about it.   We could say: okay, we've got all these  different things: a, b, c, d, e, f, g, h, j.  
Let's put them all into a single vector  and then let's create a single matrix  
that has alpha, alpha, alpha, alpha;  beta, beta, beta, beta, et cetera.  
And then if we do this matrix  multiplied by this vector we get this.
With these grey zeros in the  appropriate places, which gives us this,  
which is the same as this. And  so this shows that a convolution  
is actually a special kind of matrix  multiplication: it's a matrix multiplication   where there are some zeros that are fixed and  some numbers that are forced to be the same.  
Now, in practice, it's going to be faster  to do it this way, but it's a useful kind of  
thing to think about, I think that just to  realize like: oh! it's just another of these  
special types of matrix multiplications.
Dropout
Okay, I think… well, let's look at one more thing.  Because there was one other thing that we saw,  
and I mentioned we would look at, in the  tabular model, which is called Dropout,  
and I actually have this in my excel spreadsheet.  If you go to the “conv-example (dropout)” page …  
you'll see we've actually got a little bit more  stuff here. We've got the same input as before   and the same first convolution as before  and the same second convolution as before.  
And then we've got a bunch of random numbers.  
They're showing as between 0 and 1, but they're  actually… (that's just because they're rounding  
off…) they're actually random numbers between,  you know, that are floats between 0 and 1.
Over here we're then saying… if…  
let's have a look… so way  up here (I'll zoom in a bit)  
I've got a dropout factor – let's  change this, say to 0.5, there we go.
So over here this is something that says:   if the random number in the equivalent  place is greater than 0.5, then 1,  
otherwise 0. And so here's a whole bunch of 1 and  0. Now this thing here is called a dropout mask.
Now what happens is: we multiply (over here), we  multiply the dropout mask and we multiply it by  
our filtered image. And what that means is we end  up with exactly the same image we started with…  
(here's the image we started with…) but it's  corrupted. Random bits of it have been deleted,  
and based on the amount of dropout we  use – so if we change it to, say, 0.2 –  
not very much of it's deleted at all,  so it's still very easy to recognize.   Or else if we use lots of dropouts, say 0.8, it's  almost impossible to see what the number was.
And then we use this as the input to the next  layer. So, that seems weird, why would we delete  
some data, at random, from our processed  image, from our activations after a layer  
of the convolutions? Well, the reason is that  a human is able to look at this corrupted image  
and still recognize it's a seven, and the idea  is that a computer should be able to as well,  
and if we randomly delete  different bits of the activations,  
each time, then the computer is forced to  learn the underlying real representation  
rather than overfitting. You can  think of this as data augmentation,   but it's data augmentation not for the inputs, but  data augmentation for the activations. So this is  
called a Dropout Layer. And so Dropout Layers  are really helpful for avoiding overfitting.
And you can decide how much you want to compromise  between good generalization – so lack of… so good,  
you know, avoiding overfitting – versus   getting something that works really well on the  training data. And so, the more dropout you use,  
the less good it's going to be on the training  data, but the better it ought to generalize.
And so this comes from a   paper by Geoffrey Hinton's group, quite a few  years ago now, Ruslan's now at Apple, I think.  
And then Krizhevsky and Hinton  went on to found Google Brain.  
And so you can see here they've got this picture  of a, like, fully connected neural network,   two layers, just like the one we built,  and here, look, they're kind of randomly  
deleting some of the activations, and all  that's left is these connections. And so,   there's a different bunch that's going  to be deleted each, each, each batch.
I thought this was an interesting  point. So Dropout, which is   super important, was actually  developed in a master's thesis  
and it was rejected from the main neural networks  conference, then called NIPS, now called NeurIPS.  
So it ended up being disseminated  through… arXiv, which is a preprint server  
and yes, it's just been pointed out on our  chat that Ilya was one of the founders of  
OpenAI. I don't know what happened to Nitish,  I think he went to Google Brain as well, maybe.
Yeah so, you know, peer review is a  very fallible thing, in both directions,  
and it's great that we have preprint  servers so we can read stuff like this,   even if reviewers decide it's not worthy, it's  been one of the most important papers ever.
Activation functions
Okay, now… I think that's given us a good tour now, we've  really seen quite a few ways of dealing with input  
to a neural network, quite a few of the things  that can happen in the middle of a neural network.   We've already talked about “Rectified  Linear Units”, which is… this one here:  
0 if “x” is less than 0 or “x”  otherwise. These are some of the other  
activations you can use. Don't use this one, of  course, because you end up with a linear model.  
But they're all just different functions.  I should mention, like, it turns out these   don't matter very much. Basically pretty much  any non-linearity works fine, so we don't spend  
much time talking about activation functions,  even in Part 2 of the course, just a little bit.  
So yeah, so we understand there are… there's  our inputs, they can be one-hot encoded,  
or embeddings, which is a compute/computational  shortcut. There are sandwich layers of matrix  
multipliers and activation functions. The matrix  multipliers can sometimes be special cases, such   as the convolutions or the embeddings. The output  can go through some tweaking, such as the Softmax,  
and then, of course, you've got the loss function  such as cross-entropy loss, or mean squared error  
or mean absolute error. But, you know, it's not…  there's nothing too crazy going on in there.
So I feel like we've got a good  sense now of, like, what goes inside,   you know, a wide range of neural nets, you're  not going to see anything too weird from here.  
And we've also seen a wide range of applications. So before you come back to do Part 2, you know,  what now? And we're going to have a little AMA  
session here and, in fact, one of the questions  was: what now? So this is quite… quite good
One thing I strongly suggest is, if  you've got this far, it's probably worth  
you investing your time in reading  Radek’s book, which is “Meta Learning.”  
And so Meta Learning is very heavily based  on the kind of teachings of fast.ai over the  
last few years and is all about how to learn  deep learning and learn pretty much anything.
Yeah, because, you know, you've got to this point,  you may as well know how to get to the next point  
as well as possible.
And…
The main thing you'll see that Radek talks  about (or one of the main things) is practicing  
and writing. So if you've kind of zipped through  the videos on, you know, 2x and haven't done any  
exercises, you know, go back and watch the videos  again. You know a lot of the best students end up  
watching them two or three times – probably more,  like three times. And actually go through and code  
as you watch, you know, and experiment. You know,  write posts, blog posts, about what you're doing.  
Spend time on the forum both helping others  and seeing other people's answers to questions.  
Read the success stories on the forum and  of people's projects to get inspiration for  
things you could try. One of the most important  things to do is to get together with other people,  
for example, you can do, you know, a Zoom study  group. In fact on our discord, which you can  
find through our forum, there's always study  groups going on, or you can create your own.  
You know a study group to go  through the book together.   Yeah, and of course, you know, build  stuff. And sometimes it's tricky to  
always be able to build stuff for work because  maybe there isn't… you're not quite in the right   area, or they're not quite ready to try out  Deep Learning yet, but that's okay. You know,  
build some hobby projects, build some stuff  just for fun, or build some stuff that you're   passionate about. Yes, it's really important to…  to not just put the videos away, and go away and  
do something else because you'll forget everything  you've learnt and you won't have practiced.
So one of our   community members went on to create  an activation function, for example,  
which is Mish, which is now, as Tanishq has just  reminded me on our forums, is now used in many of  
the state-of-the-art networks around the  world, which is pretty cool… this is… and,  
he's now at Mila, I think, a research… one of  the top research labs in the world. I wonder  
how that's doing? Let's have a look… go to Google  Scholar… nice, 486 citations. They're doing great.
All right, let's have a look at how  our AMA topic is going and pick out  
Jeremy AMA
some of the highest ranked AMA’s.
How do you stay motivated?
Okay.
So, the first one is from Lucas and… actually  maybe I should… actually let's switch our  
view here. So, our first AMA is from Lucas.  And Lucas asks: “How do you stay motivated?  
I often find myself overwhelmed in this field,  there are so many new things coming up that  
I feel like I have to put so much energy  just to keep my head above the waterline…”
Yeah, that's a very interesting question, I mean,  I think Lucas… the important thing is to realize:  
you don't have to know everything, you know. In  fact nobody knows everything, and that's okay.  
What people do is they take an interest  in some… some area… and they follow that  
and they try and do their best, the best job  they can of keeping up with some little sub  
area. And if your little sub area is too  much to keep up on, pick a sub sub area. 
Yeah, there's nothing like… there's no need  for it to be demotivating that there's a   lot of people doing a lot of interesting  work and a lot of different subfields.  
That's cool, you know. It  used to be kind of dull with,   you know, there were only basically five  labs in the world working on neural nets.
And yeah, from time to time, you know, take a dip  into other areas that maybe you're not following   as closely. But when you're… but when you're  just starting out you'll find that things are  
not changing that fast at all really, it can kind  of look that way because people are always putting  
out press releases about their new tweaks, but  fundamentally the stuff that is in the course  
now is not that different to what  was in the course five years ago:  
the foundations haven't changed. And  it's not that different, in fact,   to the Convolutional Neural Network that  Yann LeCun used on MNIST back in 1996.  
It's, you know, the basic ideas I've described are   forever, you know, the way the inputs work  and the sandwiches of matrix multipliers and  
activation functions and the stuff you do to the  final layer, you know. Everything else is tweaks.  
And the more you learn about those basic ideas,  the more you'll recognize those tweaks as  
simple little tricks that you'll be  able to quickly get your head around. So then Lucas goes on to ask… or to comment…  “Another thing that constantly bothers me is  
Skew towards big expensive models
I feel the field is getting more and more skewed  towards bigger and more computationally expensive   models and huge amounts of data. I keep wondering  if in some years from now I would still be able  
to train reasonable models with a single GPU or if  everything is going to require a compute cluster.”
Yeah, that's a great question, I get that a lot.  But interestingly, you know, I've been teaching  
people machine learning and data  science stuff for nearly 30 years  
and I've had a variation of this question  throughout. And the reason is that  
engineers always want to push the envelope in,  like, the… on the biggest computers they can find,  
you know. That's just this like fun thing  engineers love to do. And, by definition, they're  
going to get slightly better results than people  doing exactly the same thing on smaller computers.  
So it always looks like: “oh you need  big computers to be state-of-the-art.”  
But that's actually never true, right, because  there's always smarter ways to do things,  
not just bigger ways to do things. And  so, you know, when you look at fastai's  
DAWNBench success when we trained ImageNet  faster than anybody had trained it before,  
on standard GPU’s, you know, me and a bunch of  students, that was not meant to happen, you know,  
Google was working very hard with their TPU  introduction to try to show how good they were,  
Intel was using like 256 PC’s  in parallel, or something…  
but yeah, you know, we used common sense and  smarts and showed what can be done. You know,  
it's also a case of picking the problems you  solve. So I would not be probably doing like,  
going head-to-head up against Codex and trying  to create code from english descriptions,  
you know, because that's a  problem that does probably   require very large neural nets  and very large amounts of data.  
But if you pick areas in different domains,  you know, there's still huge areas where  
much smaller models are still  going to be state-of-the-art.
So hopefully that helped answer your question.  All right, let's see what else we got here.
How do you homeschool children
So Daniel has alway been following my journey  with teaching my daughter math, yeah he's  
so: “I homeschool my daughter.” And  Daniel asks: “how do you homeschool   young children science in  general and math in particular?  
Would you share your experiences by blogging  or in lectures someday.” Yeah, I could do that.  
So I actually spent quite a few months just  reading research papers about education  
recently. So I do probably have a lot I  probably need to talk about at some stage.  
But yeah, broadly speaking, I lean into  
using computers and tablets,  a lot more than most people,   because actually there's an awful lot of really  great apps that are super compelling – they're  
adaptive, so they go at the right speed for  the student, and they're fun, and I really like  
my daughter to have fun, you know, I really  don't like to force her to do things.  
And for example there's a really cool app called  DragonBox Algebra 5+ which teaches algebra to  
five-year-olds, by using a really fun computer  game involving helping dragon eggs to hatch.  
And it turns out that yeah, algebra, the basic  ideas of algebra, are no more complex than the  
basic ideas that we do in other kindergarten  math and all the parents I know of who have  
given their kids DragonBox Algebra 5+, their  kids have successfully learned algebra.  
So that would be an example, but yeah, we  should talk about this more at some point.
All right, let's see what else we've got here.
Walk-through as a separate course
So Farah says: “The walk-thrus  have been a game changer for me.  
The knowledge and tips you shared in those  sessions are skills required to become an   effective machine learning practitioner  and utilize fastai more effectively.  
Have you considered making the walk-thrus a  more formal part of the course, doing a separate   software engineering course or continuing  live-coding sessions between Part 1 & 2?”
So yes, I am going to keep  doing live-coding sessions.   At the moment we've switched to those,  specifically to focusing on APL,  
and then in a couple of weeks they're  going to be going to fast.ai study groups,   and then after that they'll gradually turn  back into more live-coding sessions. But yeah,  
the thing I try to do in my live-coding, or study  groups, whatever, is yeah, definitely try to show  
the foundational techniques that just make  life easier as a coder or a data scientist.  
When I say foundational I mean, yeah this, the  stuff which you can reuse again and again and   again, like learning regular expressions  really well, or knowing how to use VIM,  
or understanding how to use the terminal and  command line, you know, all that kind of stuff.  
Never goes out of style, it never  gets old. And yeah, I do plan to,  
at some point, hopefully, actually do a course  really all about that stuff specifically,  
but yeah, for now the, for now the best approach  is follow along with the live-coding and stuff.
How do you turn model into a business
Okay wgpubs, which is Wade, asks: “How  do you turn a model into a business?  
Specifically, how does a coder with  little or no startup experience   turn an ML based gradio prototype  into a legitimate business venture?”.
Okay, I plan to do a course about  this, at some point, as well. So,  
you know, obviously there isn't  a two-minute version to this, but   the key thing with creating a legitimate business  venture is to solve a legitimate problem, you  
know, a problem that people need solved/solving.  And which they will pay you to solve.
And so, it's important not to start with your  fun gradio prototype as the basis your business,  
but instead, start with: here's a problem  I want to solve. And generally speaking  
you should try to pick a problem that you  understand better than most people. So it's  
either a problem that you face day to day in your  work, or in some hobby or passion that you have,  
or that, you know, your club has or your local  school has, or your… your spouse deals with in  
their workplace, you know. It's something where  you understand that there's a… something that  
doesn't work as well as it ought to. Particularly  something where you think to yourself: “you know,  
if they just used Deep Learning here, or some  algorithm here, or some better compute here,  
that problem would go away.” And  that's, that's the start of a business.  
And so then, my friend Eric Ries wrote a book  called “The Lean Startup” where he describes  
what you do next, which is basically,  you fake it. You create, so he calls it,   the “minimum viable product”. You create something  that solves that problem, it takes you as little  
time as possible to create. It could be very  manual, it can be loss making, it's fine.  
You know, even the bit in the middle where you're  like: “oh there's going to be a neural net here…”   it's fine to, like, launch without the  neural net and do everything by hand.
You're just trying to find out: “are people going  to pay for this?” and is this actually useful?   And then once you have, you know, hopefully  confirmed that the need is real, and that people  
will pay for it, and you can solve the need, you  can gradually make it less and less of a fake,  
you know, and to you know more and more  getting the product to where you want it to be.
Jeremy's productivity hacks
Okay, I don't know how to pronounce  the name m-i-w-o-j-c, m-i-w-o-j-c  
says: “Jeremy, can you share  some of your productivity hacks?  
From the content you produce that  may seem you work 24 hours a day?”
Okay, I certainly don't do that. I think  one of my main productivity hacks actually  
is not to work too hard. Or at least… no,  not to work too hard… not to work too much.  
I spend, probably, less hours a day  working than most people, I would guess,  
but I think I do a couple of things differently  when I'm working. One is I've spent half,  
at least half, of every working day, since I was  about 18, learning or practicing something new.
Could be a new language, it could be a new  algorithm, it could be something I read about.  
And, nearly all of that time,  therefore, I've been doing that thing  
more slowly than I would if I  just used something I already knew  
– which often drives my co-workers crazy because  they're like, you know: “why aren't you focusing  
on getting that thing done”. But in the other  50% of the time I'm constantly, you know,  
building up this kind of exponentially improving  base of expertise, in a wide range of areas. And  
so now I do find, you know, I can do things,  often, orders of magnitude faster than people  
around me or certainly many multiples faster  than people around me because I, you know, know  
a whole bunch of tools and skills and ideas which  (yeah/no) other people don't necessarily know. So  
like I think that's one thing that's been helpful.  And then another is, yeah, like trying to really  
not overdo things, like get good  sleep, and eat well, and exercise well.
And also I think it's a case of like,   tenacity, you know, I've noticed a lot of people  give up much earlier than I do. So yeah, if you,  
if you just keep going until something's actually  finished then that's going to put you in a small  
minority, to be honest, most people don't do that.  When I say finish, like finish something really  
nicely. And I try to make it like… so I  particularly like coding and so I try to  
do a lot of coding related stuff.  So I create things like nbdev,   and nbdev makes it much much easier for me  to finish something nicely, you know. So,  
in my kind of chosen area I've spent quite a bit  of time trying to make sure it's really easy for  
me to like, get out a blog post, get out a  Python library, get out a notebook analysis,  
whatever. So yeah, trying to make these things I  want to do easier and so then I'll do them more.
Final words
So, well, thank you everybody. That's been a  lot of fun. Really appreciate you taking the  
time to go through this course with me. Yeah,  if you enjoyed it, it would really help if you  
would give a like on youtube, because it really  helps other people find the course… goes into  
the youtube recommendation system. And please do  come and help other beginners on forums.fast.ai.  
It's a great way to learn yourself… is to try to  teach other people. And yeah, I hope you'll join  
us in Part 2. Thanks very much… thanks everybody,  very much. I really enjoyed this process and  
I hope to…, I get to meet more of  you in person in the future. Bye.
