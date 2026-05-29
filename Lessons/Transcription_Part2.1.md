# 9

Lesson 9: Deep Learning Foundations   to Stable Diffusion, 2022 https://youtu.be/ZkOtBv123V0 Hi everybody and welcome to “Deep Learning  Foundations to Stable Diffusion”. Hopefully  
it's not too confusing that this is described  here as Lesson 9, that's because strictly  
speaking we treat this as Part 2 of the  “Practical Deep Learning for Coders” series.  
So that Part 1 had eight lessons so this is lesson  nine. But don't worry you didn't miss anything,  
it's the first lesson of Part 2, which is called  “Deep Learning Foundations to Stable Diffusion”  
and maybe rather than calling it “Practical  Deep Learning for Coders” we should call this   “Impractical Deep Learning for Coders” in the  sense that we are certainly not going to be  
spending all of our time seeing exactly how to  do important things with with deep learning.  
But we'll be doing a whole lot of fun things  —generative modely fun things— and also a  
whole lot of understanding lots of details  which you won't necessarily need to know  
to use this stuff but, if you want to become a  researcher or if you want to, like, put something  
in production —which has got like some kind  of complex customization requirements, stuff   like that, then it is going to be very helpful  to learn the details we'll be talking about.
So here in Lesson 9, there's kind  of going to be two parts to it. One  
is just a quick run through —quickish  run through of using Stable Diffusion—  
Because we're all dying to  play with it, right? And then  
the other thing that I'll be doing is describing  in some detail what's going on, how is it working.
There'll be a whole lot of hand waving,  either way, because it's going to take   us a few lessons to describe everything  from scratch but hopefully, you know,  
you'll get a, you'll feel like you'll come away  from this, with this lesson with a reasonable,   you know, intuitive understanding, at  least, of how how this is all working.
Assumptions. Well, I'm going  to try to explain everything,  
like, everything. I'm going to try to  explain everything so, like, if you haven't  
done Deep Learning before this is going to be very  hard but I will at least be trying to say like,  
this is roughly what's going on  and where you can find out more. Having said that, I would strongly suggest  doing Part 1 before doing this course unless  
you really want to throw yourself in the deep,  deep end and give yourself quite, quite a test.  
If you haven't done Part 1 of “Practical Deep  Learning for Coders”. But you're, you know,   reasonably comfortable with Deep Learning basics,  you could like, write a basic SGD loop in Python.  
And you know, you know, how to use, ideally  Pytorch —but Tensorflow is probably okay as well.  
And you kind of know the basic ideas of how to  create, you know, what an embedding is and you  
could create one of those from scratch, stuff  like that, you know, you'll probably be fine. Generally speaking, for these courses,  I find most people tend to watch the  
videos a few times and often the second  time through folks will like pause and   look things up they don't know and check  things out. Generally speaking, you know,  
we expect people to be spending about 10 hours  of work on each video. Having said that some  
people spend a hell of a lot more and go very  deep, some people will spend a whole year,   you know, sabbatical studying “Practical  Deep Learning for Coders” in order to really  
fully understand everything so really  it's up to you as to how deep you go. Okay, so, with that said, let's jump into  it. As I said the first part we're going  
to be playing around with Stable  Diffusion and I, you know, try to,  
you know, prepare this as late as  possible so it wouldn't be out of date.  
Unfortunately, as of 12 hours ago, it is now  out of date, and this is one of the big issues  
with the bit I'm about to describe which  is like: how to play with Stable Diffusion,   exactly how do the details work. Which is, it's  moving so quickly that all of the details I'm  
going to describe to you today and all of  the software I'm going to show you today,   by the time you watch this, if you're watching  it —so what is it we're up to now, it's 11th of  
October— so if you're watching this in, like,  December of 2022, or watching this in 2023,  
the details will have changed. So what's  happened today in the last 24 hours is: two  
papers have come out so what I was going to  be telling you today is for example to do a  
Stable Diffusion generative model the number of  steps required has gone down from a thousand to  
about 40 or 50, but then as of, yeah, last  night, papers just come out that's saying  
it's now down to 4, that is, 256 times faster.  And another paper has come out with a separate  
—I think— orthogonal approach, which makes  it another, let's see, 10 to 20 times faster.
So things are very exciting. Things are moving  very quickly. Now, having said that, don't worry  
because after this lesson we're going to be going  from the foundations —which means we're going to  
be learning all the things of how these are built  up— and those don't change much at all. And in  
fact a lot of what we'll be seeing is extremely  similar to another course we did in 2019,   because the foundations don't change and once  you know the foundations these kinds of details  
about the stuff that you'll find in these papers,  you'll be like: “oh, I see, they did all these   things the same way as usual and they made this  little change”. So that's why we do things from  
the foundations, so that you can keep up with  the research, do your own research by taking  
advantage of this foundational knowledge which,  you know, all these papers are building on top of.
So anyway, I guess I should  apologize that even as I record this,   you know, the notebook is now one day out of date.
This course vs DALL-E 2
So in Part 1, you know, you might remember we saw  this stuff of Dall-e 2 illustrations of Twitter,  
by Twitter bios, which really are pretty cool.  
So, you know, the cool thing is that we're now  at a point we can build this stuff ourselves,  
and run this stuff ourselves. We won't actually  be using this particular model Dall-e 2, we’ll be   using a different model: Stable Diffusion. But has  very similar kind of outputs. But we can go even  
further now, so, one of our wonderful alumni,  Alon, actually recently started a new company  
called —I don't know who said— it's strumer  (strmr.com) where you can use something that   we'll be learning about today, called Dreambooth,  to put any object, person, whatever into an image.  
And so he was kind enough to do a quick Dreambooth  run for me and added these various pictures of me  
using his service. So here's a fun service you  can try. One crazy one he tried was me as a dwarf  
which, I gotta say, actually worked pretty  well. This half looks like me, I reckon,  
and then the bottom bit is the dwarf version.  So thank you a lot and congratulations on  
your great progress since completing  the fastai course. Yeah, I love it.
So, something that's a bit different about  this compared to previous courses —a lot  
of previous courses— is this is no longer  just a me thing because this is moving so  
quickly I've needed to get a lot of help to even  vaguely get up at the gate and stay up to date.  
So everything I'll be showing you today  is very heavily influenced by extremely  
high levels of input from these amazing  folks —all of whom are fast AI alumni.  
So Jonathan Whitaker —who I saw in our chat— is,  was basically the first guy to create detailed  
educational material about Stable Diffusion  and has been in the generative model space,  
well, for a long time by Stable  Diffusiony standards —I guess.  
So Wasim um has been an extraordinary  contributor to all-things fastai.  
Petro came to San Francisco for the last  time we did a Part 2 course in 2019 and  
took what he learned there and made his amazing  Camera+ software dramatically better and had it  
highlighted by Apple for the extraordinary Machine  Learning stuff added and he's now at Hugging Face   working on the software that we'll be using a  lot —diffusers— and then Tanishq —everybody in  
the forum of the fastai community probably  already knows— now at Stability.Ai working  
on Stable Diffusion models, his expertise  particularly as in medical applications.
So really folks from all the key groups  —pretty much— around Stable Diffusion and  
stuff are working on this together. And you'll  also find some of these folks have recorded  
additional videos going into more detail about  some of the areas which you'll find on the,  
on the course website. So make sure you go  to course.fast.ai to get all the information  
How to take full advantage of this course
about, you know, all the materials that you need  to take full advantage of this. So every lesson  
has links to notebooks and to details and  so forth. If you want to go even deeper,  
head over to forums.fast.ai,  enter the Part 2 2022 category,  
hit on the “about the course” button and  you'll find that every lesson there's a chat  
with even more stuff. So look at this carefully  to see all the things that me and the community  
have provided to you to help you understand this  video and also check out the questions underneath  
and answers underneath to see what people have  talked about. They can get a bit overwhelming so  
once they get big enough they'll, you'll see  that there's a summarize button that you can,  
that you can click to kind of see just the  most liked parts. So that can be very helpful.
Okay so, they're all important resources, I think,   to get the most, to get the  most out of this course.
Cloud computing options
Now, compute, so I'm completing Part 2  requires quite a bit more compute than Part 1.  
Compute options are changing rapidly and, to be  honest, the main reason for that is because of   the huge popularity of Stable Diffusion. Everybody  is taken to using Colab for Stable Diffusion and  
Colab's response has been to start charging by  the hour for most usage. So you may well find if  
you're a Colab user —we still love Colab. You may  find that, you know, you run out of, they start  
not giving you decent GPUs anymore and if you want  to then upgrade, they limit quite a lot how many  
hours you can use. So at the moment, yeah, still  try Colab, they're pretty good, I mean, for free  
you get some decent stuff, but I would strongly  suggest trying out also Paperspace Gradient. You  
can pay like nine dollars a month to actually  get some pretty good GPUs there at the moment.  
Or I'll pay them a bit more to get even better  ones. Again, but the thing is this is all going  
to change a lot, I don't know, like, maybe people  will make Paperspace Gradient have to change their  
pricing too, I don't know, so check course.fast.ai  to find out what our current recommendations are.
Now, Lambda Labs and Jarvis Labs are also both  good options. Jarvis was created by alum of the  
course and has some just really fantastic  options at a very reasonable price. And a  
lot of fastai students use them and love them. And  also check out Lambda Labs who are the most recent  
provider on this page and they are rapidly adding  new features. But the reason I particularly  
wanted to mention them is, at least as I say  this, which as I say is early October 2022,  
they're the cheapest provider of kind of big  GPUs that you might want to use to run like  
serious models. So they're absolutely,  yeah, well worth checking out. But as I say,  
this could all have changed by the time you  watch this, so go and check out course.fast.ai. Also, at the moment: late 2022, GPU prices  have come down a lot and you may well want  
to consider buying your own machine at this point.
Okay so what we're now going to  do is jump into the notebooks  
Getting started (Github, notebooks to play with, resources)
and so there's a repo that we've  linked to called “diffusion-nbs”  
which isn't kind of the main course notebooks  —it's not the “From the Foundations Notebooks…”—  
there's just a couple of notebooks that you  might want to play with, a bit of fun stuff to   try out. One of the interesting things here is…  Jonathan Whitaker —who I tend to call Johno so,  
if I say Johno, that's who I'm referring to—  has created this really interesting thing   called suggested_tools.md which hopefully he'll  keep up to date so, even if you come here later,  
this will still be up to date. Because he knows  so much about this area he's been able to pull out  
some of the best stuff out there for just starting  to play. And I think it's actually important to  
play because that way you can really understand  what the capabilities are and what the constraints  
are so that then you can think about like, well,  what could you do with that and also, like, what  
kind of research opportunities might there be.  So I'd strongly suggest trying out these things.
The community on the whole has moved towards  
making things available as Colab notebooks, so  if I click, for example, on this one: Deforum.  
And they often have this kind of hacker  aesthetic around them, which is kind of fun.  
So what happens is like just they add  lots and lots of features and you can  
basically just fill in this stuff to try… to  try things. And they often have a few examples  
and so you can hit up the Runtime and say:  Change Runtime Type to make sure it says GPU,  
and you can say what kind of GPU,  and start running things. Now,  
a lot of the folks who use this stuff honestly  have no idea what any of these things mean,  
now, by the end of the course you all know what  all of these things mean, pretty much. And that  
will help you to make great outputs from stuff  like this. But you can create out great outputs  
just using more of an artisanal approach,  there's lots of information online about,  
you know, what kinds of things could you try.  So, anyway, check out this stuff from from Johno.  
And then he also links to this fantastic  resource from pharmapsychotic which  
is rather overwhelming list of things to play  with. Now again, you know, maybe by the time you  
watch this this is all changed but I just wanted  to know these kind of things are out there and  
they're basically like ready to go applications  that you can start playing with. So play a lot.
What you'll find is that most of them —at least at  the moment— expect you to input some text to say  
what you want to create a picture of. It turns out  that, as we'll learn, we'll learn in detail why,  
the text you pick it's not very easy to know what  to write and and that gives kind of interesting  
results. At the moment it's quite an artisanal  thing to understand what to write and the best  
way to learn what to write —is called the The  Prompt— the best way to learn about prompts is to   look at other people's prompts and their outputs.  So at the moment perhaps the best way to do that  
is Lexica which has lots and lots  of really interesting artworks.  
And so, AI artworks, and so you can click  on one and see what prompt was used.  
And so you'll see here that, generally, you start  with what do you want to take… make a picture of,  
what's the style and then the  trick is to add a bunch of, like,  
artists names or places that they put art  so that the algorithm will tend to create a  
piece which matches art, you know, that tends to  have the these kinds of words in their captions.  
So there's a really useful trick to, kind  of, get good at this. And so you can even  
search for things —so I don't know  if they have teddy bears, let's try—  
there we go, so if there's a kind of like a  —probably not that one— that's a pretty good  
teddy bear image. So you can kind of get some  sense of how to create nice teddy bear images.  
So cute —I know what I'm going to be showing  my daughter tomorrow. You can see they often  
tend to have similar kinds of stuff to try to  encourage the algorithm to give good outputs. Okay so by the end of this  course you'll understand  
why this is happening: why these kinds  of out, you know, prompts create these   kind of outputs and also how you can go  beyond just creating prompts to actually,  
to actually, building really innovative  new things with new data types.
Diffusion notebook from Hugging Face
Okay. So let's take a look at the  diffusion-nbs repo, the first thing  
we'll look at is stable_diffusion(.ipybn)  So a couple of options here. You can,  
you can clone this repo which is linked from  both the course.fast.ai and from the forum  
and run it on like Paperspace Gradient or your own  machine or whatever. Or you can head over to Colab  
and you can just say GitHub, right, and then you  can paste in the link to it directly from GitHub.
Okay, so, I'm running it on my own machine  
and this notebook is largely been built thanks to  the wonderful folks at Hugging Face and Hugging  
Face have a library called Diffusers, so any  of you that have done Part 1 of the course  
will be very familiar with Hugging Face.  We used a lot of their libraries in Part 1.   Diffusers is their library for doing Stable  Diffusion. And stuff like Stable Diffusion.  
At the moment, you know, these things  are changing a lot but, at the moment,   this is our recommended library for doing this  stuff and it's what we'll be using in this course.  
Maybe by the time you watch this there'll  be lots of other options, so again, keep an   eye on course.fast.ai. In general Hugging  Face have done a really good job of being  
at and staying at the kind of head of the pack  around models in general for Deep Learning so,  
that would be, you know, not surprising if they  continue to be the best option for quite a while.  
But the basic idea of any library  is going to look pretty similar. So, to get started playing with this  you will need to log in to Hugging Face.  
So if you've got a Hugging Face you can create  a username there and a password and then login.  
Once you've done it once it'll save it on your  computer so you won't have to log in again.  
And the thing we're going to be working with is  pipelines and in particular the Stable Diffusion  
Pipeline. Again, you know, that you might be  using different pipelines by the time that  
you watch this. But the basic idea of pipeline is  quite similar to what we call a Learner in fastai,  
which is it's got a whole bunch of  things in it, you know, a bunch of,   kind of, processing and models and inference,  all happening automatically. And just like you  
can save a pipeline in fastai, sorry, save  a Learner in fastai, you can save a pipeline  
in Diffusers. Now something that you can do  in all pretty much all Hugging Face libraries,  
you can't do in fastai, is you can then save a  pipeline —or whatever— back up into the cloud  
onto Hugging Face —they call it the Hub—  and so then if we say from_pretrained(),   it's a lot like how we create pre-trained Learners  in firstai but the thing you put here is actually  
—if it's not a local path— it's a Hugging Face  repo. So if we search Hugging Face for this,  
and you can see   this is what it's going to download. And you can  actually save your own pipelines up to the hub  
for other people to use. So I think this is a very  nice feature that helps, you know, the community  
build stuff. So this is actually going to… the  first time you run this, it's going to download  
many gigabytes of data from the internet. This is  one of the slight challenges with using this on  
Colab is every time you use Colab everything  gets thrown away and start from scratch so,   it'll all have to be downloaded every time  you use Colab. If you use something like  
Paperspace or particularly actually Lambda  Labs it's all going to be saved for you.
So once you've downloaded all this it's going  to be, it's going to save a whole bunch of stuff  
into your dot cache in your home directory, so  that's where Hugging Face puts things. So now  
that we have a pipeline called pipe we can now  treat it as if it's a function —which is pretty  
common for like Pytorchy stuff and fastai stuff,  you should be very familiar with this hopefully. And you can pass it a prompt. And so this  is just some text. And that's going to  
return some images. Since we're only passing  one prompt it's going to return one image,   so we'll just index into dot images. And  when we run it, it takes uh, you know,  
maybe 30 seconds or so and returns “a  photograph of an astronaut riding a horse”.  
Every time you call a pipeline using the  same random seed you'll get the same image.  
You can send them at the random seed manually, and  so you could send to somebody else and say: “oh!,  
this is a really cool astronaut riding a horse,  I found, try manual seed 1024. And you'll get  
back this particular astronaut riding a horse.  So that's how you can, like, that's the most  
basic way to get started running on Colcab or on  your own machine, you can start creating images.
It takes, as I said, takes 30 seconds or so and in  this case it took 51 steps. What it's doing, this  
How stable diffusion works
is a bit very different to like what we're used to  with inference in fastai, where it's one step to  
classify something, for example. What it's doing  in these 51 steps is: it's starting with like,  
so this is actually an example that we're going  to create ourselves, ourselves in the course of  
creating handwritten digits, and this is actually  an image from a later notebook we'll be building.   Well it basically starts with random noise and  each step it tries to make it slightly less noisy  
and slightly more like the thing we want. And  so going down here is showing all the steps to  
create the first four, for example, or here to  create the first one. And if you look closely,  
you can kind of see, in this noise, there is  something that looks a bit like a one and so it   kind of decides to focus on that. And so that's  how these diffusion models, basically, work.  
So remember, if you're having any trouble  finding the materials we're looking at to,   go to course.fast.ai or go to the forum topic  to see all the links. And this one is called  
diffusion dash nbs, and the notebook is called  —you can see it at the top— stable_diffusion.
Now, a question might be, well why don't  we just do it in one go, and we can do it  
in one go but if we try to do it in one go it  doesn't do a very good job. These models aren't,  
as I speak now in October 2022, smart enough  to do it in one go. Now as I mentioned,  
at the start, the fact that I'm doing it in 51  steps here is, you know, hopelessly out of date  
because as of yesterday, apparently, we can now  do it in three to four steps —I'm not sure if   that code's available yet— so by the time you see  this, yeah, this might all be dramatically faster,  
but as I'll be describing, understanding  this basic concept —I'm pretty confident   it's going to be very important, like, forever—  so we'll talk about that. So if we do 16 steps  
instead of 51 steps, you know, it looks a  bit more like it but it's still not amazing.
Okay, so, that's how you can kind of get started  and I'll show you a few things that you can tune  
and, you know, I should remind you that, you  know, we're not… most of the stuff I'm showing   you in this was built by Pedro Cuenca  and the other folks at Hugging Face so,  
huge thanks to them, there's no way I could  have been as up to up to speed with all this  
detail without their help. They built this library  Diffusers and have done a fantastic job of helping  
display what you can do with it. So let's look at  an example of what you can do with it, we're just   going to quickly define a little function here to  create a grid of images —the details don't matter.  
Diffusion notebook (guidance scale, negative prompts, init image, textual inversion, Dreambooth)
But what we do want to show here is: you  can take your prompt, which is “an astronaut  
riding a horse” and just create four copies  of it. Okay so, times, when applied to a list,  
simply copies the list that many times, so here's  a list of the exact same prompt, four times.  
And then what we're going to do is we're going  to pass to the pipeline the prompts and we're   going to use a different parameter now, called  “guidance scale”. We're going to be learning  
about “guidance scale”, in detail, later in  the course but basically, what this does is,  
it says to what degree should we be, kind of,  focusing on the specifics caption versus just  
creating an image. So we're going to try a  few different guidance scales, about 1, 3,   7, 14. Generally seven and a half, I believe, at  this stage is the default, that may have changed  
by the time you watch this. And so each  row here is a different “guidance scale”.  
So you can see, in the first row, it hasn't  really listened to us very much at all, these  
are very weird looking things in there, none of  them really look like “astronauts riding a horse”.  
At guided scale of 3, they look more like  things riding horses, that they might be  
astronautish and at 7.5 they're certainly, on  the whole, look like astronauts riding a horse  
and at 14 or 15 they certainly look like that  but they're getting a little bit too abstract  
sometimes. I have a pretty strong feeling there  are some slight problems with actually how this  
is coded, or actually how the algorithm works  which I will be looking at during this course  
so maybe, by the time you see this, some  of these will be looking a bit better.  
I think basically something that's happening  here is it's actually, kind of, over…   over jumping a bit too far during these high ones.
Anyway, so, the basic idea of what it's doing here  is, this guidance is, it's basically actually,  
for every single prompt, it's creating, creating  two versions: one version of the image with the  
prompt —“an astronaut riding a horse”— and  one version of the image with no prompt. So  
it's just some random thing, and then it takes  the average, basically, of those two things.  
And that's how, that's what “guidance  scale” does, and you can kind of think   of the “guidance scale” as being a bit like  a number that's used to weight the average.
There's something very similar you can do, where  again, you create, get the model to create two  
images but, rather than taking the average, you  can ask it to effectively subtract one from the  
other. So, here's something that Pedro did  of using the prompt a “Labrador in the style  
of Vermeer”. And then he said, well, what if we  then subtract something which is just the model  
for the caption blue. And you can pass in this  thing “negative prompt” to diffusers and what  
that will do is it will take the prompt, which in  this case is “Labrador in the style of Vermeer”,  
and create a second image, effectively, which  is just responding to the prompt “blue” and  
effectively subtract one from the other, the  details are slightly different to that but that's   the basic idea, and that way we get a non-blue  “Labrador in the style of Vermeer”. So, yeah, this  
is the basic kind of idea of how to use negative  prompt, and you can play with that, good fun.
Here's something else you can play with, is  you don't have to just pass in text, you can  
actually pass in images. So for this you'll need a  different pipeline, you'll need an image to image  
pipeline. And with the image to image pipeline  you can grab a rather sketchy looking sketch  
[Laughter] and you can then pass to  this eye to eye image to image pipeline  
the initial image to start with. And basically  what this is going to do is, rather than starting  
diffusion process with random noise,  it's going to basically start it with  
a noisy version of this drawing and so  then it's going to try to create something  
that matches this caption, and also, like,  follows this kind of guiding starting point.  
And so as a result, you get things that look  quite a lot better than the original drawing  
but you can see that the composition is the  same and so, using this approach, you can,  
you know, construct things that match the  particular kind of composition you're looking for.  
So I think that's quite a nifty approach. And  so here this parameter “strength” is saying:  
to what degree do you want to really  create something that looks like this or,  
to what degree do you want the model to be able  to, you know, try out different things, a bit.
Now, here's where thing get interesting and this  is the kind of stuff you're not going to be able   to do at the moment with just the basic GUI's and  stuff but, you can, if you really know what you're  
doing, what we could do now is we could take these  output images and we could say: “oh, this one's  
nice”, sorry, this one. This one's nice, let's  make this the initial image. And now we'll say:  
Let's do an oil painting of… by Van Gogh. And  passing the same thing here, and a strength of 1.  
And actually that pretty much worked. And I think  that's absolutely fascinating, right? because  
this is something I haven't seen before —which  Pedro put together this week— and it's combining  
simple python code together.  And so you can play with that. 
Something else you can do —which this one's  actually example came from the folks at Lambda   Labs— is, and we won't be going into this  in detail right now because this is like,  
basically, exactly like what we've done a  thousand times in fastai, is you can take   the models in that pipeline and you can pass  it your own images, and your own captions.  
And so what happened here is…  (oh I hate these things, go away,  
never mind) Here we are. This one, so, what these  folks did —I think this was Justin, if I remember  
correctly— yeah, so, what Justin at Lambda did  was he created a really cool data set by going to  
grab a Pokémon dataset of images, which had  almost a thousand images of Pokémon. And then,  
this is really neat, he then used an imaging  captioning —image captioning— model to  
automatically generate captions  for each of those image, images.   And then he fine-tuned the Stable Diffusion  model using those image and caption pairs.  
So here's an example of one of the captions in  one of the images. And then took that fine-tuned  
model and passed it prompts like “Girl with  a girl with a pearl earring” and “Cute Obama  
creature” and got back these Totoro, these (Oopsy  Daisy!)... And got back these super nifty images  
that now are reflecting the fine tuning dataset  that he used and also responding to these prompts.  
Here's another example of something you can  do. Fine-tuning can take quite a bit of data  
and quite a bit of time but you can actually do  some special kinds of fine-tuning. One that you  
can do is called Textual Inversion which is where  we actually fine-tune just a single embedding. So,  
for example, we can create a new embedding where  we're trying to make things that look like this.  
So what we can do is we can  give this concept a name.  
So here we're going to call  it… Oh, I just lost it now.  
Watercolor. There we are. We're going  to call it “watercolor-portrait”.  
And so that's what the embedding name we're  going to use is. And we can then basically  
add that token to the text model and then we can  train the embeddings for this so that they match  
the example pictures that we've seen. And  this is going to be much faster because we're  
just trading a single token for just, in this  case, four pictures. And so when we do, that  
we can then say, for example, “woman  reading in the style of” and then  
passing that token we just trained, and  as you see, we'll get back a kind of novel  
image which I think is, yeah,  pretty, pretty interesting.
Another example, very similar to textual  inversion, is something called Dreambooth  
which, as mentioned here, what it does is it  takes an existing token but one that isn't used  
much like, say, “sks” —nothing, almost nothing has  “sks”— and fine-tunes a model to bring that token,  
as it says here, close to the images we  provide, and so what Pedro did here was he  
grabbed some pictures of me and said “painting  of sks —so this case he's fine-tuned this token  
to be Jeremy Howard photos— in the style of Paul  Signak”, and there they are. And so the example  
I showed earlier of the dwarf Jeremy Howard, that  service strmr (Astria - https://www.strmr.com/) is   actually using this, Dreambooth. So  here's how you can try that yourself.
Okay so, that is part one of this lesson  which is the: how to get started playing  
around with Stable Diffusion. In  part two we're going to talk about  
what's actually going on here from a machine  learning point of view. So we'll come back in  
about seven minutes to talk about that. All  right, see you guys in about seven minutes.
Okay, welcome back folks. I just thought I'd share  with you one more example, actually of textual  
inversion training this is my daughter's teddy:  Tiny —who as you can see is grossly misnamed.  
And Pedro and I tried to, yeah, create a textual  inversion version of Tiny and I was trying to get  
Tiny riding a horse and it's interesting that,  when I tried to do that, this top row here —this  
is actually Pedro's example when he ran it— this  is showing the kind of steps as he was training,  
of trying to use the caption “Tiny riding a  horse”. And as you can see it never actually  
ended up generating Tiny riding a horse,  instead it ended up generating a horse  
that looks a little bit like  Tiny. And then we're trying to get   “Tiny sitting on a pink rug” and actually, after a  while it did make some progress there, it doesn't  
quite look like Tiny. One thing Pedro did that  was different to me was he started with the  
embedding of a person —in my one actually  started with the embedding for Teddy and   it worked a bit better. But, as you see, there  are like problems and we'll understand where  
those problems come from as we talk more about  how this is trained in the rest of this lesson.
Okay so I'm going to be relying on some  understanding of the basic idea of how  
machine learning models are trained here so,  if you start getting a bit lost at any point  
you might want to go back to Part 1 and  then come back to this once you're unlost.  
The way we're going to start… okay so I  need to explain, the way Stable Diffusion  
Stable diffusion explained
is normally explained is focused very much  on a particular mathematical derivation.  
We've been developing a totally new  way of thinking about Stable Diffusion  
and I'm going to be teaching you that. It's  mathematically equivalent to the approach which  
you'll see in other places but what you'll realize  and discover is that it's actually, conceptually  
much simpler. And also later in this course we'll  be showing you some really innovative directions  
that this can take you when you think of it in  this brand new way. So, all of which is to say,  
when you listen to this, and then you go and  look at some blog post and it looks like I'm   saying something different, just keep that  in mind. I'm not saying something different,  
I'm expressing it in a different way but  it's, it's equally mathematically valid.
What I'm going to do is I'm going to  start by saying: let's imagine that…  
let's imagine that we were trying to get something  to generate something much simpler which is to  
generate handwritten digits, okay. So it's like  the Stable Diffusion for handwritten digits.  
And we're going to start by assuming there's some  like API, some web service or whatever out there  
—who knows how it was made?— but what it does  is something pretty nifty which is that you can  
get an image of a handwritten digit and you can  pass it over into this… this web API into this,  
you know, this REST endpoint or whatever, it's  just a black box as far as we're concerned,  
and it's going to spit out the probability that,  this thing you passed in, is a handwritten digit.  
So for this one so, let's  say this image is called X1,  
the probability that X1 is a handwritten  digit, it might say is, 0.98.  
And so then you pass something else into this  magic API endpoint, which looks like this.  
You pass that in. And that looks a  little bit like an eight, I guess,   but it might not be, you'd pass it into  this API and see what happens. This is X2  
and it says the probability  that X2 is a digit is 0.4.  
Okay, now we pass in our image X3…  
into our magic API and it returns the probability  
that X3 is a handwritten digit… pretty small.
Okay so, why is this interesting.  Well, it turns out that if you have  
a function, you know, let's not call this  an API, let's call this, let's call this,  
it's called “f”, it's some function, but  it's like behind some web API REST endpoint,  
whatever. If you have this function we can  actually use it to generate handwritten digits.  
So that's something pretty magical and  we're going to see how on earth would   you do that. If you have this function which  can take an image and tell you the probability  
that that is a handwritten digit, how could you  use it to generate new images? Well, imagine you  
wanted to turn this mess into  something that did look like  
an image. Here's something you could do: let's  say it's a 28 by 28 image which is, what? 786…  
Oopsy Daisy, 28 times 28: 784. So 784 pixels and we could  pick one of these pixels  
and say: what if I increase this pixel to be a  little bit darker? And then we could pass that  
image through “f” and we could see what happens  to the probability that it's a handwritten digit.  
So, for a specific example, handwritten  digits don't normally have any  
pixels that are black in the very  bottom corners so, if we took this here,   and we said: what would happen if we made  this a little bit lighter? Right? And then  
we pass that exact image through here. The  probability would probably go up a tiny bit.  
For example. So now, we've got an image which is  slightly more like a handwritten digit than before  
and, also, in digits generally, there  are straight lines, so this pixel here,  
it probably makes sense for it to be darker. So if  we made a slightly darker version of this pixel,  
and sent it through here, that would also  increase the probability a little bit.  
And so we could do that for every single  pixel of the 28 by 28, one at a time,   finding out which ones: if we make them a little  bit lighter, make it more like a handwritten  
digit, which ones: if we make it a little bit  darker, make it more like a handwritten digit.  
What we've just done is we've calculated  the gradient of the probability that X3  
is a handwritten digit with  respect to the pixels of X3.
Now, notice that I didn't say “dp(X3) /  dX3” —which you might be familiar with from,  
from high school— and the reason  for that is that we've done,   we've calculated this for every single pixel. And  so, when you do it for lots of different inputs,  
you have to turn the “d” into a —this  is called a del or a nabla— and it just  
means there's lots of values here. So this  here contains lots of values which is the:  
how much does the probability that X3 is a digit  increase as we increase this pixel value, and as  
we increase this pixel value, as we increase  this pixel value. So there's going to be, for  
28 by 28 inputs, there's going to be 784 pixels  which means that this thing here has 784 values.  
Math notation correction
Okay, so, with those 784 values they tell us  how can we change X3 to make it look more like a  
digit and so, what we can then do is we can now  change the pixels according to this gradient.  
And so we can do something a lot like  what we do when we train Neural Networks,   except instead of changing the weights in  a model we're changing the inputs to the  
model. And so we're going to take every pixel and  we're going to modify it: subtract its gradient  
—a little bit times its gradient— so I multiply  this by some constant, let's call it C,  
and then we're going to subtract it to  get some new image. So with the new image,  
it's probably going to get rid of  some of these bits at the bottom,   right? And it's probably going to add a few  more bits between some of these here, right?  
And we've now got something that looks slightly  more like a handwritten digit than before.  
And this is the basic idea, we can now  do that again, we can now take this we  
can run it through “f” and so we've now  got something like, say we call it X3’,  
for example, so this new version X3’ or  whatever it's now the probability that's  
a handwritten digit —it's quite a bit  higher— I'd say it's probably like 0.2,   maybe. And we can now do the same thing, we can  say for every pixel, if I increase its value a  
little bit, or decrease its value a little bit,  how does it change the probability that this  
new X3 —whatever prime prime— is a digit.  And so we'll now get a new gradient, here,  
784 values, and we can use that to change every  pixel to make it look a little bit more like  
a handwritten digit. So, as you can see, if we have this magic  function, we can use it to turn any arbitrary  
noisy input into something that looks like a  valid input, something that has a high P value  
from that function, by using this derivative. So  a key thing to remember here is this saying… is  
saying: as I change the input pixels, how  does it change the probability that this  
is a digit? And that tells me which pixels to  make darker and which pixels to make lighter.
Now, those of you who remember your high school  calculus, may recall that when you do this by  
changing each pixel one at a time to calculate a  derivative, this is called the finite differencing  
method of calculating derivatives, and it's very  slow because we have to call —find out diff, uh,  
sorry— I can't spell diff-e-ren-cing… It's very  slow because we have to call it seven of this  
function, 784 times every single one. We  don't have to use finite differencing.  
Assuming the folks running this magic API endpoint  used Python, we can just call f.backward().  
And then we can get X3.grad(), and that will  tell us the same thing in one go, by using  
the analytic derivatives. So we'll learn exactly  about what these dot backward does, we'll write  
our own everything from scratch including our own  calculus things from scratch, later, but for now,   just like we did in Part 1 of the course,  we're just going to assume these things exist.  
So maybe then the nice folks that provide  this endpoint could actually provide a new   endpoint that calls dot backward for us,  and gives us dot grad, right? And then we  
don't really have to use “f” at all, right? We  can instead just directly call this endpoint  
or just directly call this endpoint that gives  us the gradient directly, we'll multiply it  
by this smaller constant C, we'll subtract it  from the pixels, and we'll do it a few times,  
making the input get larger and larger P values,  larger and larger probabilities that this is,  
actually, a digit. So we don't particularly need  
this thing at all, we don't particularly need the  thing that calculates these probabilities. We only   need the thing that tells us how… which pixels  we should change to calculate the probabilities.
Okay, so, that's great. The problem  is: nobody's provided this for us,  
so, we're going to have to write  it. So how are we going to do that?  
Well, no problem, generally speaking,  in this course, when there's some magic  
black box that we want to exist and it doesn't  exist we create a Neural Net and we train it.  
So we want to train a Neural Net that tells us  which pixels to change to make a digit look, well,  
to make an image look more like a handwritten  digit. Okay so here's how we can do that, we could  
create some training data and use that  training data to get the information we  
Creating a neural network to predict noise in an image
want, we could pass in something that  looks a lot like a handwritten digit,  
we could pass something that looks a bit like a  handwritten digit, we could pass something in that  
doesn't look very much like a handwritten  digit, and we could pass in something which  
doesn't really look like a handwritten digit at  all. Now, you'll notice it was very easy for me  
to create these, I created real handwritten digits  and then I just chucked random noise on top of it.  
It's a little bit awkward for us to come up  with an exact score saying: how much is that  
like a handwritten digit? How much is that like  a handwritten digit?, how much is that, and how   much is that? It seems a bit arbitrary, so let's  not do that. Let's use something which is kind of  
like the opposite, right? and instead let's say:  oh, why don't we predict how much noise I added,  
right? Because this number seven is actually  equal to this number seven plus this noise.  
And this number three is actually equal  to this number three plus this noise.  
And this number six is actually equal  to this number six plus this noise.  
And that one's got a lot. And of  course the very first one is equal to  
this number nine plus this noise.  
So, why don't we generate this data and then  rather than trying to come up with some arbitrary  
number of like how much like a digit is it, let's  say the amount of noise tells us how much like a  
digit it is. So something with no noise is  very much like a digit and something with  
lots of noise isn't much like a digit at all.  So let's feed in, let's create a Neural Net,  
who cares what the architecture is, right?  It's just a Neural Net of some kind.  
And this is critical to your understanding  of this course, at this point, we're going  
to go beyond the idea of like worrying all  the time about architectures and details  
and we're going to be spending, quite  often, we're going to get, I mean,   we're going to get to all those details  but the important thing to using this stuff  
well is to think about Neural Nets as being  something that has some inputs, some outputs,  
—Oopsy Daisy.   Some outputs. And some loss  function which takes those two  
and then the derivative is  used to update the weights,  
right? That's really what we care about: those  four things. Now the inputs to our model is this.
Okay, that's the inputs to our model. The outputs  to our model is a measure of how much noise there  
is. So maybe we could just say: Oh, well, what's  the… these are all basically normally distributed  
random variables, with a mean of zero, and a  variance, in this case, of zero. In this case  
they're normally distributed random variables  with mean of zero and a variance of like 0.1.   This one's normally distributed random variables  —pixels, I guess— with a mean of zero and like  
0.3. This one's super noisy. There's the mean  and variance. So that's the mean for each one  
and the variance for each one. So, why  don't we, as the output, use the variance.  
So predict how much noise or better still,  why don't we predict the actual noise itself.  
So why don't we actually use  
that. Now we're not just predicting how  much noise but we predict the actual noise.  
That's our outputs. Now if we do that  our loss is going to be very simple,  
it's going to be: we took the input,  we passed it through our Neural Net,  
we tried to predict what the noise was.  And so the prediction of the noise is  
n hat and the actual noise is n and so we  can do something we've done a thousand times,  
which is we can divide it by the count,  squared, and then we can sum all that up  
and this here is the Mean Squared  Error, which we use all the time.  
So the Mean Squared Error means that we've  now got inputs, which is noisy digits,  
we've got outputs, which is noise, and so  this Neural N etwork is trying to predict  
this noise. So we're basically jumping straight  to the step that we had here —remember this is  
what we really wanted. We wanted some ability to  know how much do we have to change a pixel by,  
to make it more digit-like.  Well, tudum! this number seven  
into this number seven —that's our goal— we have  to remove all of that. So if we can predict the  
noise, then we've got exactly what we want,  which is this: we can then do this process,  
we can take multiply it by a constant and subtract  it from our input, and so, if you subtract this  
noise from this input you get this handwritten  digit. So we're doing exactly what we wanted.  
Well that seems easy enough, we already know  from Part 1 how to do this. So we just have  
any old Neural Network, so some ConvNeXt, or  something that takes as input numbers where we've  
just randomly added different amounts of noise,  lots of noise to some, not much noise to others,  
it predicts what the noise was that we added,  we take the loss between the input, sorry,  
between… we take the loss between the predicted  output and the actual noise, Mean Squared Error,  
and we use that to update the weights. And so  if we train this for a while, then if we pass  
this into our model, it will return that. And  
we're done, we now have  something that can generate  
images. How? because now we can take this trained  Neural Network —so I'm going to copy it down here—  
and we can pass it something very,  very, very noisy —which is pure noise.  
We pass it to the Neural Net and it's going to  spit out information saying which part of that  
does it think is noise, and it's going to leave  behind the bits that look the most like a digit,  
just like we did back here. So it might  say: oh!, you know what? if you left behind  
just that bit, that bit, that bit, that bit,  that bit, that bit, that bit, and that bit,   it's going to look a little bit more like  a digit and then maybe you could increase  
the values of that bit, that bit, that  bit, that bit, that bit and that bit.   And so after you do that —and so that everything  else is noise, so we subtract those bits,  
subtract it times some constant,  we're now going to have  
something that looks more like a digit, which  is what we hoped for. And so then we can just  
do it again. And you can see now why we… you can  see now why we are doing this multiple times.
Somebody on the chat saying  they don't see me drawing,  
oh you can see, thanks Jimmy. Son't know  Michelangelo what's happening for you.
Okay.  
And to answer your earlier question  about how am I drawing, I'm using a   Graphics Tablet —which I'm not very expert  at because on Windows you can just draw  
directly on the screen, which is  why this is particularly messy—.  
All right, in practice, at the moment  —this might change by the time you've  
watched this— we use a particular type  of Neural Net for this. The particular   kind of Neural Net we use is something that was  developed for medical imaging called the U-Net.  
If you've done previous versions of the course,  you'll have seen this, and don't worry this   course we'll see exactly how our U-Net works  and we'll build them ourselves from scratch.  
And this is the first component of  Stable Diffusion, it's the U-Net.
Okay so, there's going to be a few pieces and  the details of why they're called these things  
don't matter too much, just yet, just take my  word for it this is their names. And the thing   that you do need to know for each thing is,  like, what's the input and what's the output.  
So the input to the U-Net —or what does it  do— the input to the U-Net is a, somewhat  
noisy, image. And when I say somewhat, it could  be not noisy at all or it could be all noise,  
that's the input. And the output is the  noise, such that if we subtract the output  
from the input we end up with the unnoisy  image, or at least, an approximation of it.  
So that's the U-Net.   Now, here's the problem —well, here's our problem—  we have —oh why do I keep forgetting this— we have  
28 times 28, 784, I should write that  down. We have in these things 784 pixels.  
And that's quite a lot. And it gets worse because  in practice we don't want to draw handwritten   digits, the thing that we'd be passing in here  is beautiful high definition photos or images of,  
Working with images and compressing the data with autoencoders
like, great paintings. And at the moment  the thing we tend to use for that is a 512  
by 512 by 3 Channel, RGB. Nice,  big image. 512 by 5 12 by 3.  
Red, Green and Blue. These are the pixels. So that  is 512 by 512. by 3, 786,432. So we've got 786,432  
pixels in here and so this is, I  don't know, some beautiful picture,  
this is my amazing portrait, Van Gogh style  in a dainty little hat. There we go. So,  
this is the beautiful painting, or an image of  it, it's… that's a lot of pixels and so, training  
this model where we put noisy versions of millions  of these beautiful images is going to take  
us an awful long time and, you know, if you're  Google, with a huge cloud of TPUs or something,  
maybe that's okay. But for the rest of us we  would like to do this as efficiently as possible.  
How could we do this more efficiently?  Well, when you think about it,  
in this beautiful picture I drew, storing  the exact pixel value of every single pixel  
is probably not the most efficient way  to store it. You know, what if instead,  
we said like, oh, you know —let's say  this is like green rushes or something—.  
It might say like, oh over here is green  and everything kind of underneath it's   pretty much the same, or, you know, maybe I'm  wearing a blue top in this beautiful portrait  
and it could kind of say like, oh, all the pixels  in here are blue. You know, you don't really have   to do everyone individually, there are faster,  more concise ways of storing what an image is.  
We know this is true because, for example,  a JPEG picture is far fewer bytes than the  
number of bytes you would get if you multiply  its height by its width by its channels. So we know that it's possible  to compress pictures.  
So let me show you a really interesting way to  compress pictures. Let's take this image and let's  
put it through a convolutional layer of stride  two. Now, if we put it through a convolutional  
layer of stride two, with six features,  with six channels, we would get back a 256  
by 256 —gosh that was a terrible attempt  at drawing a square, wasn't it?— 256  
by 256 —actually do it here— by, okay  let's double the number of channels to six,  
by six. And then let's put it through another  stride 2 convolution —and remember we're going  
to be seeing exactly how to do all these  things and building them all from scratch,   so don't worry if you're not sure what a stride  two convolution exactly is— and just do it again  
to get 128 by 128, and again, let's double the  number of channels. And then let's do it again,  
another stride two convolutions. So we're  just building a Neural Network here.  
So now we're down to 64 by 64 by 24, okay,  and then now let's put that through a few,  
like, Resnet blocks, to kind of squish down  the number of channels as much as we can,  
so it'll be now down to,  let's say, 64 by 64 by four.  
Okay, so here's a Neural Network.   And so the number of pixels in this  version is now 64 times 64 times 4, 16,384.  
So there's 16,384 pixels here. Okay so,  we've compressed it from 786,432 to 16,384,  
which is a 48 times decrease.
Now that's no use if we've lost our image  so, can we get the image back again?  
Sure, why not? What if we now create,  
a kind of, an inverse convolution which does  the exact opposite. So, actually let's put it  
over here. So we're going to take our 64 by 64 by  4 image, put it through an inverse convolution.  
So let's put it, let's keep  moving this over further  
back to 128 by 128 by 12. And put it  through another inverse convolution,  
so these are all, just basically, they're  just Neural Network layers. 256 by 256 by 6.  
And then finally —wrap you out—  
all the way back to 512  
by 512 by 3. Okay, we could put this whole thing  
inside a Neural Net, here's  our single Neural Network.  
And what we could do is we  can start feeding in images.   It goes all the way through this Neural Network  and out at the other end comes back, well,  
initially it's random, so initially  comes out of this is random noise.  
512 by 512 well, because I draw it inside here so  inside here initially it's going to give us random  
noise. And so now we need a loss function,  right? So the loss function we can create  
could be, to say, let's take this output and this  input and compare them and create and do an MSE,  
Mean Squared Error directly on those two pieces.  So what would that do if we train this model?  
This model is going to try to put an image through  
and going to try to make it so that what comes  out the other end is the exact same thing  
that went into it.  
That's what it's going to try to do,  because if it does that successfully  
then the Mean Squared Error would be zero.  
So I see some people in the chat saying  that this is a U-Net, this is not a U-Net,   okay? We'll get to that later, there's no  cross connections. It's just a bunch of  
convolutions that decrease in size, followed by  a bunch of convolutions that increase in size. 
And so we're going to try to train this model  to spit out exactly what it received in.  
And that seems really boring, what's  the point of a model that only learns  
to give you back exactly what came in? Well,  this is actually extremely interesting. This  
kind of model is called an auto encoder. It's  something that gives you back what you gave it.  
And the reason an auto encoder is interesting  is because we can split it in half.  
Let's grab just this bit.  
Okay, let's cut it up, let's grab that just  that bit, and then we'll cut a second half  
—okay they're not quite halves, but you know what  I mean— which is just this bit. And so let's say  
I take this image and I put it  through just this first half,   this green half which is called the encoder. Okay,  I can take this, this thing that comes out of it  
and I could save it and the thing that I'm  going to save is going to be 16,384 bytes.  
I started with something that was 48  times bigger than that (786,432 bytes)  
and I've turned it into something that's 16,384  bytes. I could now attach that to an email, say,  
or whatever, and I've now got something that's  48 times smaller than my original picture.  
So what's going to happen the person who receives  these 16,384 bytes. Well, as long as they have a  
copy of the decoder, on their computer, they can  feed those bytes into the decoder and get back  
the original image. So what we've just done  is we've created a compression algorithm.  
That's pretty amazing, isn't it? And in fact  these compression algorithms work extremely  
extremely well. And notice that we didn't train  this on just this one image, we've trained it on,  
say, millions and millions of images, right? And  then so, you and I both need to have a copy of  
these two Neural Nets, right? But now we can share  thousands of pictures that we send each other by  
sending just the 16,384 byte version. So we've  created a very powerful compression algorithm.
And so maybe you can see where this is  going. If this here is something which  
contains all of the interesting and useful  information of the image in 16,384 bytes,  
why on earth would we train our U-Net  with 786,432 pixels of information.  
And the answer is: we wouldn't. That would  be stupid. Instead we're going to do this  
Explaining latents that will be input into the unet
entire thing using our encoded version of each  picture. So if we want to train this U-Net on  
10 million pictures, we put all 10 million  pictures through the auto encoder’s encoder,  
so we've now got 10 million of these smaller  things. And then we feed it into the U-Net  
training hundreds or thousands of times to train  our U-Net. And so what will that U-Net now do?  
Something slightly different to what we described,  it does not anymore take a somewhat noisy image,  
instead it takes a somewhat noisy -one of these-.  So it'd probably help to give this thing a name,  
and so the name we give this thing is “Latents”.  These are called the Latents. Okay, so instead,  
the input is somewhat noisy Latents. The output  is still the noise and so we can now subtract  
the noise from the slightly noise somewhat noisy  Latents and that would give us the actual Latents.  
And so we can then take the output of the  U-Net and pass it into our auto encoder’s  
decoder, sorry, uh yes: decoder, because that's  something which, okay, that's something which  
takes Latents and turns it into a picture.  So the input to this is small Latents tensor  
and the output is a large image.
Okay. Now. This thing here is not going to  be called an encoder, it's going to have the  
name the VAE, and we'll learn about why later,  those details aren’t too important, but let's  
put its correct name here the VAE's decoder.  
So you're only going to need the encoder for the  VAE if you're training a U-Net. If you want to   just do inference, like we did today, you're  only going to need the decoder of the U-Net.  
So this whole thing of Latents is entirely  optional, right? This thing we described before  
works fine. But, you know, generally speaking we  would rather not use more compute than necessary  
so, unless you're trying to sell the world a  room full of TPUs, you would probably rather,  
everybody was doing stuff on the  thing that's 48 times smaller.   So the VAE is optional but it saves us a whole lot  of time and a whole lot of money. So that's good.
Okay, what's next? Well there's something else  which is we have not just been in this morning's,  
Adding text as one hot encoded input to the noise and drawing (aka guidance)
you know, it's in… sorry in the first half  of today's lesson we weren't just saying:  
produce me an image, we were saying: produce me  an image of Tiny the teddy bear riding a horse.  
So how does that bit work. So the way  that bit works is actually, on the whole,  
pretty straightforward. Let's think about how  we could do exactly that for our MNIST example.  
How could we get this, so that, rather than just  feeding in noise and getting back some digit;  
how do we get it to give us a particular  digit. What if we wanted to pass in   the literal number “3” plus some noise and have  it attempt to generate a handwritten three for us.  
How would we do that? Well, what we could do  is, way back here, for the input to this model,  
why don't is, in addition to passing  in the noisy input, let's also pass in  
a one-hot encoded version of what digit it is.  
So we're now passing two things into this  model. So previously, this Neural Net  
took his inputs: just the pixels,  
but now it's going to take in the pixels and what  digit is it as, like, a one-hot encoded vector.  
So now it's going to learn how  to predict what is the noise,  
right? And it's going to predict what is the noise  and it's going to have some extra information,  
which is, it's got a know what the original  image was so we would expect this model to  
be better at predicting noise than the previous  one because we're giving it more information.   This was a 3, this was a 6, this was a 7. So  this Neural Net is going to learn to estimate  
noise better by taking advantage of the fact  that it knows what actual the input was.
And why is that useful? Well the reason that's  useful is because now, when we feed in the number  
3, like the actual digit 3 as a one hot-encoded  vector plus noise, after this has been trained,  
then our model is going to say: the noise is  everything that doesn't represent the number 3,  
because that's what it's learned to do, right?  So that's a pretty straightforward way to give it  
—and the word we use is: guidance— about what it  is that we're actually trying to remove the noise  
from. And so then we can use that guidance to  guide it as to what image we're trying to create.  
So that's the basic idea. Now, the problem is, if  we want to create a picture of a cute teddy bear,  
How to represent numbers vs text embeddings in our model with CLIP encoders
we've got a problem. It was easy enough to  pass the digit 8, the literal the literal  
number eight into our Neural Net because we can  just create a one-hot encoded vector in which  
position number eight is a one and everything else  is a zero, but how do we do that for a cute teddy?  
We can't. We can't create every possible  sentence that can be uttered in the whole  
world and then create a one hot-encoded version,  one-hot encoded version of every sentence in  
the world because that's going to take a  vector that is too long, to say the least.  
So we have to do something else  to turn this into an embedding,  
something other than grabbing a one-hot  encoded version of this, so what do we do?
So, what we do.  
So what we're going to do is  we're going to try to create  
a model that can take a sentence like a cute teddy  
and can return a vector of numbers that in some  way represents what cute teddies look like.  
And the way we're going to do that is we're  first going to surf the Internet and download  
images, so here are four examples of  images that I found on the internet,  
and so for each of these images they  had an image tag next to them, right?  
And if people are being good then they also  added an ALT tag to help with accessibility  
and maybe for SEO purposes and they  probably said things like “a graceful swan”.  
And the ALT tag for this might have been  “a scene from Hitchcock's the birds”.  
And the ALT tag for about this  might have been “Jeremy Howard”.  
And the alt tag for this might  have been “fast.ai's logo”.  
And we could do that for millions and millions and  millions of images that we find on the internet.  
So what we can now do with these is we can create  two models. One model, which is a text encoder,  
and one model, which is an image encoder.  
Okay. So again, these are Neural Nets, we  don't care about what their architectures  
are or whatever. We know that they're  just black boxes which contain weights,  
which means they need inputs and outputs and  a loss function and then they'll do something.  
Once we've defined inputs and outputs and a loss  function, the Neural Nets will then do something.
So here's a really interesting idea.  
What if we take this image,  
and what if we then also take  the text “a graceful swan”.  
Okay, and we're going to feed these into  their respective models, which initially they,  
of course, have random weights and that  means that they're going to spit out  
random features: a vector of stuff, random  crap. Because we haven't trained them yet, okay?  
And we can do the same thing with a scene from  Hitchcock, we pass the scene from Hitchcock   in and we'll pass in the words “scene from  Hitchcock” and then we'll give us two other  
vectors, right? And so we can do  something really interesting now.  
We can line these up —I guess we'll  just move them—. We can line these up.  
Okay, here's all of our images, okay?  And then we can have —Oopsy Daisy—.  
Okay, and then we can have our text.  
So we've got “graceful swan”,  
we've got “Hitchcock”,  
we've got “Jeremy Howard”,  and we've got fastai logo.  
CLIP encoder loss function
Now, ideally, when we pass the  “graceful swan” through our model,  
what we'd like is that it  creates a set of embeddings   that are a good match for  the text “graceful swan”.  
When we pass the scene from Hitchcock through our  image model we would like it to return embeddings  
which are similar to the embeddings for the text  seen from “Hitchcock” and ditto for the picture of  
Jeremy Howard versus the name “Jeremy Howard” and  ditto for the image fastai and (sorry). The fastai  
logo and the word “fastai logo”. So, in other  words, for this particular combination here,  
we would like this one's features and  this one's features to be similar.  
So, how do we tell if two sets of  things are similar? Two vectors?  
Well, what we can do is we can simply multiply  them together, element wise, and add them up.  
And this thing here is called the dot product.  
And so we could take the dot product  of the features from the image model   for this one and the dot product of the  features from the text model for the word  
“graceful swan” and take that dot product.  And we want that number to be nice and big.  
And the scene from Hitchcock's features should be  very similar to the text scene from Hitchcock's  
features, so we want their product to be nice and  big. And ditto for everything on this diagonal.  
Now, on the other hand a graceful swan  picture should not have embeddings that   are similar to the text “a seen from Hitchcock”.  
So that should be nice and small. And  ditto for everything else off diagonal.
And so perhaps you can see where this  is going. If we add up all of these,  
right? Add those all together  and then subtract all of these:  
we have a loss function.   And so, if we want this loss function to be good,  then we're going to need the weights of our model,  
for the text encoder, to spit out  embeddings that are very similar  
to the images that they're paired with. And  we need them to spit out embed… features,  
for things that they are not  paired with, which are not similar.   And so if we can do that then, we're going to  end up with a text encoder that we can feed in  
things like “a graceful  swan”, “some beautiful swan”,  
“such a lovely swan”, and these should all give  very similar embeddings, because these would all  
represent very similar pictures. And so what  we've now done is we've successfully created  
two models, that together, put text and images  into the same space. So we've got this multimodal  
set of models, which is exactly what we wanted.  So now, we can take our cute teddy bear,  
feed it in here, get out some  features and that is what we will use  
instead of these one-hot encoded vectors when  we train our photo, or painting, whatever  
U-Net. And then we can do exactly the same  thing with guidance, we can now pass in  
the text encoder’s feature vector for a cute  teddy, and it is going to turn the noise into  
something that is similar to things that  it's previously seen that are cute teddies.  
So the model that's used, or the pair of  models that's used here, is called CLIP.  
This thing where we want these to be bigger and  these to be smaller is called a contrastive loss.  
And now you know where the CL comes from.  So here we have a CLIP text encoder.  
Its input is some text. Its output is, we  call it an embedding, it's just some features.  
Oops, embedding,   where similar sets of text, meaning, with similar  meanings, will give us similar embeddings.
Okay —because we need a bit  more space— we're nearly done.  
Okay, so we've got the U-Net that can  denoise latents into unnoisy latents,  
including pure noise. We've got a decoder  that can take latents and create an image.   We've got a text encoder which can allow us  to train a U-Net which is guided by captions.  
So the last thing we need is the question about:  how exactly do we do this inference process here?  
So, how exactly, once we've got something that  gives us the gradients we want… and by the way,  
these gradients are often called the “score  function”, just in case you come across that,  
that's all that's referring to. Yeah,  how exactly do we go about this process?  
And, unfortunately, the language used  around this is weird and confusing  
Caveat regarding "time steps"
and so, ideally, you will learn to ignore  the fact that the language is weird and  
confusing. And in particular the language  you'll see a lot talks about “time steps”.  
And you'll notice that during our training process  we never used any concept of “time steps”. This is  
basically an overhang from the particular way in  which the math was formulated in the first papers.  
There are lots of other ways we can  formulate it and during the course,   on the whole, we will avoid using the term “time  steps”. But we can, to see what time steps are,  
even though it's got nothing to do with  time in real life, consider the fact that we  
used varying levels of noise, some things were  very noisy, some things were not noisy at all,  
some things had no noise and some I haven't  drawn here would have been pure noise. You could basically create a  kind of a noising schedule.  
Where along here, you could put,  say, the numbers from 1 to a 1,000,  
and you could then say: oh, you  know —and we'll call this “t”—   and maybe we've randomly pick a number from 1 to a  1,000, and then we look up on this noise schedule  
—there should be some monotonically decreasing  function— and we'd look up, let's say we happen to  
pick randomly, a number 4, we would look up here  to find where that is and we'd look over here and  
this would return to us some Sigma, which is the  amount of noise to use if you happen to get a 4.  
So if you happen to get a 1 you're  going to get a whole lot of noise,   and if you happen to get a 1,000  you're going to have hardly any noise.  
So this is one way of, like, picking… so remember  we were going to pick in, when we were training,  
we were going to pick for every image a random  amount of noise. So this would be one way to   do that, is to pick a random number from 1  to a 1,000, look it up on this function and  
that tells us how much noise to use. So this  “t” is what people refer to as the time step.  
Nowadays you don't really have to do that that  much and a lot of people are starting to get rid   of this idea altogether and some people instead  will simply say how much noise was there. Normally  
we would think of using Sigma for standard  deviations of gaussians or normal distributions,  
but actually, much more common is to use the  Greek letter Beta. And so if you see something  
talking about Beta, they're just saying: oh, for  that particular image, when it was being trained,  
what standard deviation of noise was being used,  basically. Slightly hand wavy but close enough.  
And so what you do, each time you're going to  create a mini batch to pass into your model,  
you randomly pick an image from your training set   you, randomly pick either an amount of noise  or, since some models you randomly pick a “t”  
and then look up an amount of noise and then  you put use that amount of noise to each one  
and then you pass that mini batch into  your model to train it. And that trains  
the weights in your model so it can  learn to predict noise. And so then,  
when you come to inference time —so inference is  when you're generating a picture from pure noise—  
you want your model, you're  basically, your model is now starting  
here, right? Which is as much noise as possible.  And so you want it to learn to remove noise,  
but what it does in practice,  as we saw in our notebook, is,  
it actually creates some hideous  and rather random kind of thing.
So in fact let's remind ourselves  
what that looked like.  
This is what I created,   when we tried to do it in one step. So  remember, what we then do is we say: okay,  
what's the prediction of the noise and then we  multiply the prediction of the noise, I said, by  
some constant —which is kind of like a learning  rate— but we're not updating weights now,  
we're updating pixels. And we  subtract it from the pixels.  
So, it didn't actually predict the image, what it  actually did was it predicted what the noise is,  
so that could then subtract that  from the image, from the noisy image,  
to give us the denoised image. And so, what  we do is we don't actually subtract all of it,  
we multiply that by a constant  and we get a somewhat noisy image.  
Why don’t we do this all in one step?
The reason we don't jump all the way to the best  image we can find is because things that look  
like this never appeared in our training set  
and so, since it never appeared in our  training set, our model has no idea   what to do with it. Our model only knows  how to deal with things that look like,  
somewhat noisy latents, and so that's  why we subtract just a bit of the noise  
so that we still have a somewhat noisy latent.   So, this process repeats a bunch of times  and questions like: what do we use for C?  
Right? And, how do we go from the prediction of  noise to the thing that we subtract? These are  
all of the things that are the the kind of the  things that you decide in the actual sampler.  
And that's used both to think about, like, how do  I add the noise? and how do I subtract the noise?
And there's a few things that might be jumping  into your head at this point, if you're anything  
like me, and one is that, gosh, this looks  an awful lot like Deep Learning optimizers.  
So in a Deep Learning optimizer this constant  
is called the Learning Rate. And we have some  neat tricks where we say, for example, oh,  
if you change the same parameters by a similar  amount, multiple times in multiple steps,  
maybe you should increase the amount you change  them, this concept is something we call Momentum.  
And we'll be doing all this from  scratch during the course, don't worry,   and in fact we even got better ways of  doing that, where we kind of say, well,  
what about what happens if the variance  changes, maybe we can look at that as well   and that gives us something called  Adam. And these are types of optimizer.  
And so maybe you might be wondering:  could we use these kinds of tricks?  
And the answer, based on our very  early research is yes, yes we can.  
The whole world of, like, where Stable Diffusion  and all these diffusion-based models came from,  
came from a very different world of maths,  which is the world of differential equations.  
And there's a whole lot of very parallel  concepts in the world of differential equations,  
which is really all about taking  these like little steps, little steps,   little steps and trying to figure out how  to take bigger steps. And so, different  
differential equation solvers use a lot of the  same kind of ideas, if you squint, as optimizers.  
One thing the differential equations solvers do,  which is kind of interesting though, is that they  
tend to take “t” as an input. And in fact, pretty  much all diffusion models —I've actually lied—,  
pretty much all diffusion models, don't just take   the input pixels and the digit, or the  caption, or the prompt they also take  
“t”. And the idea is that the model will  be better at removing the noise if you tell  
it how much noise there is. And remember,  this is related to how much noise there is.
I very strongly suspect that this premise is  incorrect, because if you think about it. for  
a complicated fancy Neural Net figuring out how  noisy something is, is very, very straightforward.  
So I very much doubt we actually need to pass  in “t”. And as soon as you stop doing that,  
things stop looking like differential equations   and they start looking more like  optimizers. And so, actually,  
Johno started playing with this and experimenting  a bit and early results suggest that, yeah,  
actually when we re-think about the whole thing  as being about learning rates and optimizers,  
maybe it actually works a bit better. In fact  there's all kinds of things we could do. Once we  
stop thinking about them as differential equations  and worry about the math —don't worry about the   math so much about gaussians and whatever— we  can really switch things around. So for example,  
we decided, for no particular obvious reason,  to use MSC. Well, the truth is, in statistics  
and Machine Learning almost every time you see  somebody use MSE it's because the math worked out   better that way. Not as in: it's a better thing  to do but, as in, you know, it was kind of easier.
Now, MSE does fall out quite nicely as being a  good thing to do with some particular premises,  
you know, like it's not like totally arbitrary  but, what if we instead used more sophisticated  
loss functions where we actually said, well, you  know, after we subtract the outputs, how good  
is this really? Does it look like a digit? or,  does it have the similar qualities to a digit?  
So, we'll learn about this stuff, but this  thing is called, for example, Perceptual Loss.  
Or another question is, do we really need  
to do this thing where we actually put noise  back at all?, could we instead use this directly?  
These are all things that suddenly become  possible when we start thinking of this  
as an optimization problem rather than a  differential equation solving problem. So,  
for those of you are interested in, kind of, doing  novel research, this is some of the kind of stuff  
that we are starting to research at the moment  and the early results are extremely positive,  
both in terms of how quickly we can do things  and what kind of outputs we seem to be getting.
Okay, so, I think that's probably  a good place to stop it. So what   we're going to do in the next lesson is  we're going to finish our journey into  
this notebook to see some of the code behind the  scenes of what's in a pipeline when I get there.  
So we'll be looking inside the pipeline  and see exactly what's going on behind   the scenes a bit more in terms of the code.  And then we're going to do a huge rewind  
to the “from the foundations” and we're going to  build up from some very tricky ground drills. Our  
ground drills would be we're only allowed to  use pure Python, the Python standard library  
and nothing else, and build up from there until  we have recreated all of this, and possibly,  
some new research directions at the same time.   So that's our goal and so strap in  and see you all next time. See ya!

---

# 9A

hello everyone my name is Jonathan and today I'm going to be taking you through this stable diffusion Deep dive notebook
looking into the code behind the kind of popular high-level apis and libraries
and tools and so on to see what exactly does the generation process look like and how can we modify that how do each
of the individual components work so feel free to run along with me if you
haven't before this might take a little while to run just because it's downloading these large models if they
aren't already downloaded and loading them up and so we're going to start by just kind of recreating what it looks
like to generate an image using say one of the existing pipelines in hugging face so we're going to basically have
Replicating the sampling loop
copied the code from the core method of the default stable diffusion Pipeline and so if you go and view that here
you'll see that we're going to be basically replicating this code but now we'll be doing it on our own sort of
notebook and then we'll slowly understand what each of these different parts is doing so we've got some set up we've got some sort of loop running
through a number of sampling time steps and regenerating an image so this is supposed to be a watercolor picture of
an otter and it's very very cool that this model can just do this but now we want to know how does that actually work
what's going on so the first component is the autoencoder now this stable diffusion is
The Auto-Encoder
a latent diffusion model and what that means is that it doesn't operate on pixels it operates on
go in the latent space of some other autoencoder model in this case a variational autoencoder that's been
trained on a large number of images to compress them down into this latent representation and bring them back up
again so I have some functions to do that we're going to look at what it's like in action by just downloading a
picture from the internet opening it up with pil so we have this 512 by 512
pixel image and we're going to load it in and then we're going to use our function defined above to encode that
into some latent representation and what this is doing is calling the vae.encode method on a tensor version of the image
and that gives us a distribution and so be sampling from that distribution and we're scaling by this because that's
what the authors of the the model did they scaled the latents down before they fit them to the model and so we have to
do that scaling and then the reverse when we're decoding just to be consistent with that but the key idea is
that we go from this big image down to this 4x64 by 64 latent representation
right so we've gone from this much larger image down and if we visualize what the four channels here and this
four different 64 by 64 channels what that looks like we'll see that it's capturing something of the image right
you can sort of see the same shapes and things there but it's not quite a direct mapping or anything for example there's
this weirdness going on the beak some of the channels look slightly Stranger than the others so there's some sort of Rich information
captured there and if we decode this back what we'll see is that the decoded
image looks really good you really have to look closely to tell the difference between our input image here and the
decoded version so very very impressive compression right this is a factor of
eight in each Dimension so 512 by 512 down 64 by 64. it's like a factor of 64
reduction in in data but it's still somehow capturing most of that information it's a very
information Rich representation and this is going to be great because now we can work with that with our diffusion model
and get nice high resolution results even though we're only working with these 64 by 64 latents now it doesn't
have to be 64 by 64. you can go and modify this to say what if this is you know 640
and encode that down and you'll see that it's just that same factor of 8 reduction
um and there we go now we have 80 by 64. and this just has to be a multiple of
eight otherwise you'll get I think an error okay so we have our encoded version of this image and that's pretty great
Adding Noise and image-to-image
the next component we're going to look at is the scheduler and I'll look more closely at this later but for now we're going to focus on this
idea of adding noise right so during training we add some noise to an image and then the model tries to predict what
that noise is and we're going to do that to different amounts so here we're going to recreate the the same type of
schedule and you can try different schedulers from the library oops and these parameters here Beta start beta
end beta schedule that's how much noise was added at different time steps and how many time steps are used during
training for sampling we don't want to have to do a thousand steps so we can set a new
number of time steps and then we'll see how these correspond with the scheduler.time steps attribute
to the original um training time steps so here we're going to have 15 sampling steps and
that's going to be equivalent to starting at times at 909 and just moving linearly down to time steps zero
we can also look at the actual amount of noise present with the sigmas attribute so again starting High moving down and
if you want to see what that schedule looks like we can plot that here and if you want to see the time steps you'll see that it's
just a linear relationship so there we go we're going to start at a
very high noise value and we're going to slowly slowly try and reduce this down until ideally we get an image out
okay so this Sigma is the amount of noise added let's see what that looks like so I'm going to start with some
random noise that's the same shape as my latent representation my encoded image and then I'd like to be equivalent to
sampling step 10 out of 15 here so I'm going to go and look up what time set that equates to and that's going to be
one of the arguments that I passed the scheduler.add noise function so I'm calling scheduler.add noise giving it my
encoded image the noise and what time step I'd like to be noising equivalent
to and this is going to give me this noisy but still recognizable version of the image and you can go and say okay
what if I look at somewhere earlier in the process right does it look more noisy what about right at the beginning
right at the end and feel free to play around there um okay so this adding noise what are we
actually doing what does the code look like let's inspect the function and you'll see that there's some setup for
different types of argument and shapes but the key line is just this noisy samples is equal to original samples
plus the noise scaled by the sigma parameter right so that's all it is it's
not always the same different papers and implementations will add the noise slightly differently but in this case
that's all it's doing so scheduler.add noise just adding noise that's the same shape as the latents scaled by the sigma
parameter um okay so that's what we're doing um so if we want to start from random noise instead of a noisy image we're
going to scale it by that same Sigma value so that it looks the same as an image that's been scaled by that amount
but then before we feed that to the actual model we then have to handle that scaling again you could do it like this
but now we have this scale model input function associated with the scheduler just to hide that complexity away
okay so now we're going to look at the same kind of sampling loop as before but we're going to start now with our image
we're going to take our encoded image we're going to noise it to some time step and then we only can denoise from
there so in code we are now preparing our text and everything the same as before which we'll look at we're setting
our number of inference steps to 50 right I'm inference steps is equal to 50 here and we're saying I'd like to start
at the equivalent of set 10 out of 50. so I'll look up what time step that equates to I'll add noise to my image
equivalent to that step and then we're going to run through sampling but this time we're only going
to start doing things once we get above that start step so I'm going to ignore the first 10 out of 50 steps and then
beyond that I'm now going to start with this noisy version of my input image and I'm going to denoise it according to this prompt
and the Hope here is that by starting from something that has some of the sort of rough structure and color of that
input image I can kind of fix that into my generation but I've got a new prompt
a National Geographic photo of a colorful dancer and here we go we see this is the same sort of thing as the
parrots but now we have this completely different actual content thanks to a different prompt and so that's a fun
kind of use of this image to image process you might have seen this for taking drawings adding a bunch of noise
and then denoising them into fancy paintings and so on so again this is something that there's existing tools
for this right the strength parameter and the image to image pipeline that's just something like this um what step
are we starting at how many steps are we skipping um but you can see that this is a pretty powerful technique for getting a bit of
extra control over like composition and color and a bit of the structure um okay so that's that's that trick with
adding noise and then using that as image to image the next big section I'd like to look at is how do we go from a
The Text Encoding Process
piece of text that describes what we want into a numerical representation that we can feed to the model
so we're going to trace out that Pipeline and along the way we'll see how we can modify that for a bit of fun
so step number one we're taking our prompt and we're turning it into a sequence of discrete tokens so here we
have in this case 77 because that's the maximum length um discrete tokens it's always going to
be that if your prompt is longer it'll truncate it and if we decode these tokens back we'll see that we have a special token
for the start of the text then a picture of a puppy and then the rest is all the same token which is this kind of end of
text padding token right so we have this special token for puppy this special
token has its own meaning end of text and the the prompts are always going to be padded to be the same length
so before in the code that we were using there we always jump straight to the so-called output embeddings which is
what we fed to the model as conditioning and so somehow this captures some information about this prompt and but
now we want to say well how do we get there how do we get from this sequence of tokens to these output embeddings
what is this text encoder um forward pass doing
right so we can look at this and there's going to be multiple steps the first is going to be some embeddings so if we
look at the text encoder.txt model.embeddings we'll see there's a couple of different ones we have token
embeddings right and so this is to take those individual tokens token 49408 or whatever and map it into a
representation that's a numerical representation so here it's a London bedding there are seven sorry about 50
000 rows one for each token and for each token we have 768 values so that's the
embedding of that token and if we want to feed one in and see what the embedding looks like here's the token for puppy and here's the token embedding
right 768 numbers they somehow capture that meaning of that token on its own and we can do the same for all of the
tokens in our prompt so we feed them through this token embedding layer and now we get 77 768 dimensional
representations of this of each token now these are all on their own no matter
where in the sentence is it is the token embedding will be the same so the next step is to add some positional
information some models will do this with some kind of like learned pattern of positioning but in this case the
positional embedding is just another London bedding but now instead of having one embedding for every token we have
one embedding for every position out of all 77 possible positions and so just like we did for the tokens we can feed
in the position IDs one for every possible position and we'll get back out an embedding for every position in the
prompt um combining them together there's again multiple ways people do this in the
literature but in this case it's as simple as adding them that's why they made them the same shape so that you can
just add the two together and now these input embeddings have some information related to the token and some related to
the position so so far we haven't seen any big model just two London beddings but this is
getting everything ready to feed through that model and so we can check that this is the same as if we just called the
embeddings layer of that model which is going to do both of those steps at once but we'll see just now why we want to
separate that out into individual ones okay so we have these individual tokens
and they have some positional information we have these final embeddings now we'd like to turn them into something that has a richer
representation thanks to some big Transformer model and so we're going to feed these through and I made this
little diagram here each token is going to turn into a token embedding get combined with the positional embedding
and then it's going to get fed through this Transformer encoder which is just a stack of these blocks and so each block
has some magic like attention has some feed forward components there's additions and normalizations and skips
and so on as well but we're going to have some number of these blocks all stacked together and the outputs of each
one get fed into the next block and so on until we get our final set of hidden States these encoder hidden States aka
the output embeddings and this is what we feed to our units to make its predictions
so the way we get this I just copied the text encoder.textmodel.forward method pulled out the relevant bits we are
going to take in those input embeddings combined positional and token embeddings and we're going to feed that through the
textmodel.encoder function with some additional parameters around attention masking and telling it that we'd like to
Output the hidden States rather than the final outputs so if we run this we can
just double check these embeddings are going to look just like the output embeddings we saw right at the beginning
so we've taken that one step tokens to output embeddings and we've broken it down into this number of smaller steps
where we have tokenization getting our token embeddings combining with position embeddings feeding it through the model
and then that gives us those final outputs so why have we gone through this trouble well there's a couple of things we can
do one demoed here I'm getting the token embeddings but then I'm looking up
um where is the token for puppy and I'm going to replace it with a new set of embeddings and this is going to be another just learned embedding of this
particular token here two three six eight right so I'm kind of cutting out the token embedding for puppy slipping
in this new set of token embeddings and I'm going to get some output embeddings which at the start look very similar to
the previous ones in fact identical but as soon as you get past the position of puppy in that prompt you're going to see
that the rest have changed right so we've somehow messed with these embeddings by slipping in this new token
embedding right at the start and if we generate with those embeddings which is what this function is doing we should
see something other than a puppy and sure enough drumroll uh we don't we get a cat and so now you
know what token two three two three six eight means um we've managed to slip in a new token
embedding and get a different image okay what can we do with this why is this fun um well a couple of Tricks first off we
could look up the token embedding for skunk right which is this number here and then instead of now just replacing
that in place of puppy what if I make a new token embedding that's some combination of the embedding
of puppy and the embedding of skunk right so I'm taking these two token embeddings I'm just averaging them and
I'm inserting them into my set of token embeddings for my prompt
um in place of just the word puppy and so hopefully when we generate with this we get something that looks a bit like a puppy a bit like a skunk and this
doesn't work all the time but it's pretty cute when it does there we go puppy skunk hybrid okay so that's not
Textual Inversion
the real reason we're looking at this the main application at the moment of being able to mess with these token
embeddings is to be able to do something called textual inversion so in textual inversion we're going to
have our prompt tokenize it and so on and but here we're going to have a special learned embedding for some New
Concept right and so the way that's trained is going to be outside of the scope of this notebook but there's a
good blog post and Community notebooks and things for doing that but let's just see this in application here so I've
um there's a whole library of these Concepts stable diffusion concept Library where you can browse through
um tons and tons and tons look over 1 400 different Community contributed
token embeddings that people have trained and so I'm going to use this one here this verb style here's some example
outputs and then these are the images it was trained on so these pretty little bird paintings done by my mother I've
trained a new token embedding they tries to capture the essence of the style and that's represented here in
this learned embeds.bin so if you download this and then upload it to wherever your notebook's running I have it here londonbeds.bin we can load that
in and you'll see that it's just a dictionary where we have one key that's the name of my new style and then we
have this token embedding 768 numbers and so now instead of slipping in the token embedding for cat we're going to
slip in this new embedding which we've loaded from the file into this prompt so masking the style of puppy tokenize get
my token embeddings and then I'm going to slip in this replacement embedding um in place of the embedding for puppy
and when we generate with that we should hopefully get a mouse in the style of this kind of cutesy watercolor
on rough paper image and sure enough that's what we get very cute little drawing of a mouse in an apron
apparently uh okay so very very cool application again there's a nice
inference notebook that makes this really easy you can say a katoya in the style of verb style you don't have to worry about manually replacing the token
embeddings yourself but it's good to know what the code looks like under the hood right how are we doing that what stage of the text embedding process
we're modifying very fun to get a bit of extra control and a very useful technique because now
we can kind of augment our model's vocabulary without having to actually retrain the model itself we're just
learning a new token embedding so very very powerful idea and really fun to play with and like I said there's thousands of community contribute
contributed tokens but you can also train your own I think I link The Notebook from here but it's also in all
the docs and so on here's the training notebook okay final little trick with embeddings
rather than messing with them at the token embedding level we can push the whole prompt through that entire process to get our final output embeddings and
we can mess with those at that stage as well so here I have two prompts a mouse and a leopard tokenizing them encoding
them with the text encoder so that's that whole process and these final output embeddings I'm just going to mix them together According to some factor
and generate with that and so you can try this with you know a cat and a snake
um and you should be able to get some really fun uh different chimeras and oops a snail apparently
okay well I can't spell um but yeah have fun with that doesn't have to be animals I'd love to see what
you create with these weird mixed up Generations okay we should look at the actual model
The UNET and classifier free guidance
itself the key unit model the diffusion model um what is it doing what is it predicting what is it accepting as
arguments so this is the kind of cool signature we call our units forward pass
and we feed in our noisy latents the time step and it's like the training
time step and the encoder hidden States right so those text embeddings that we've just been having fun with so doing
that without any Loops or anything I'm sitting in my scheduler getting my time step getting my noisy latent and my text
embeddings and then we're going to get our model prediction and we'll look at the shape of that and you'll see that this prediction has the same shape as
the latents and given these noisy latents what the model is predicting is the noise
component of that and actually it's predicting the noise component scaled by Sigma so if we
wanted to see what the original image looks like we could say well the the denoise latents is going to be the
current noisy latents minus Sigma times the model prediction right
um and so when we're denoising we're not going to go straight to that upward prediction we're going to just remove a little bit of the noise at a time but it
might be useful to visualize what that final prediction looks like so that's what we're doing here making a folder to
store some images preparing our text scheduler and input and then we're going to do this Loop but now we're going to
get the model prediction and instead of just updating our latency by one step we're also going to store an image all
right I'm decoding these two images an image of the predicted completely denoised like original sample so that's
this predicted original sample here you could also calculate this yourself latent 6 error is equal to the current latents minus Sigma times the noise
prediction right so those two should work equivalently but this Loop is going to
run and it's going to save those images to the steps folder which we can then visualize and so once this finishes in a
second or two on the left we're going to see the kind of noisy input to the model at each stage and on the right we're
going to see the noisy inputs minus the noise prediction right so the denoised version and so just give it a second or
two to run it's taking it a little bit longer because it's decoding those images each time saving them
um but once this finishes we should have a nice little preview video
okay here we go so this is the noisy latent and if we take the model's noise prediction and subtract it from that we
get this very blurry output and so you'll see as we play this oh I've left some modifications in from
last time sorry um when you see this this guidance scale we'll be back at uh I think it was eight
um in the next section we'll talk about classified free guidance and so I've been modifying that example my bad I
might cut this out of the video we'll see so I've got to wait a few seconds again for that to generate
and I'll do so as patiently as I can
okay so here we go again the noisy inputs the predicted denoise version and
you can see at the start it's very blurry but over time it gradually converges on our final output
um and you'll notice that on the left these are the latents as they are each step they don't change particularly
drastically a little bit at a time but at the start when the model doesn't have much to go on its predictions do change
quite a bit at each step right it's much less well defined and then as we go forward in time it gets more and more
refined better and better predictions and so it's got a a more accurate estimation of the noise to remove and we
remove that noise gradually until we finally get our output quite fun to visualize the process and
hopefully that helps you understand why we don't just make one prediction and do it in one step right because we get this
very blurry mess but instead we do this kind of iterative um sampling there which we'll talk about
very shortly before then though the final thing I should mention classify free guidance what is that
well like you saw when I accidentally generated the version with a much lower guidance scale
um the way classifier free guidance works is that in all of these Loops we haven't actually been passing one set of
noisy latents through the model we've been passing two identical versions and
as our text embeddings we've not just been passing the embeddings of our prompts right these ones here we've been
concatenating them with some unconditional embeddings as well and what the unconditional embeddings are is just a blank prompt right no text
whatsoever so just all padding passing that through so when we get our predictions here we've given in two sets
of latents and two sets of text embeddings we're gonna get out two predictions for the noise so we'll be
splitting that apart one prediction for the unconditional like no prompt version and one for the prediction based on the
prompt and so what we can do now is we can save all my final prediction is going to be the unconditional version plus the
guidance scale times the difference right so if you think about it if I predict without the noise I'm predicting
here if I predict with the noise sorry with the text encoding with the prompt I
get this prediction instead and I'd like to move more in that direction I'd like to push it even further towards the
prompt version and and Beyond and so this guidance scale can be larger than one to push it even more in that
direction and this it turns out is kind of key for getting it to follow the prompt nicely and I think it was first
um brought up in the Glide paper AI coffee break on YouTube has a great video on that but yeah really useful
trick or really neat hack depending on who you talk to but it does seem to work and the higher the guidance scale the
more the model will try and look like the prompts kind of in the extreme versus the lower guidance scale it might
just try and look like a generic good picture okay we've been hiding away some
Sampling explanation
complexity in terms of this scheduler.step function so I think we're going to step away from The Notebook now
and scribble a bit on some paper to try and explain exactly what's going on with sampling and so on and then we'll come
back to the notebook for one final trick all right so here's my take on sampling
and to start with I'd like you to imagine the space of all possible images so this is a very large High dimensional
space for 256 by 256 by 3 image that is 200 000 dimensional and my paper
unfortunately is only two dimensional so we're going to have to squish this down a fair bit and use our imagination
now if you just look at a random point in this space this is most likely not going to look like anything recognizable
it'll probably just look like gobbled noise but if we map an image into this space
um we'll see that it has some sort of fixed point and a very similar image almost pixel equivalent it's going to be
very close by now there's this theory that you'll hear talked about called manifold Theory
which says that for most real images like a data set of images these are
going to lie on some lower dimensional manifold within this higher dimensional space right in other words if we map a
whole bunch of images into the space they're not going to fill the whole Space they're going to be kind of clustered onto some surface now I've
drawn it as a line here because we stuck with 2D but this is a much higher dimensional plane equivalent okay so
each of these ones here is some image and the reason that I'm starting with
this is because we'd like to generate images we'd like to generate plausible looking images not just random nonsense
and so we'd like to do that with diffusion models so where did they come in well we can start with some image here
some real image from our training data and we can push the way from the manifold of like plausible existing
images by corrupting it somehow so for example just adding random noise that's equivalent to like moving in some random
Direction in this space of all possible images and so that's going to push the image away
and then we can try and predict using some model what this noise looks like right how do I go from here back to a
closable image what is this noise that's been added and so that's going to be our our big unit that does that prediction
that's going to be our diffusion model great and so that's um in in this language going to be
called something like a score function right how do I get from wherever I am what's the noise that I need to remove to get back to a plausible image
um okay so that's all well and good um we can train this model with a number of examples because we can just take our
training data add some random noise predict predict try and predict the noise update on model parameters so we
can hopefully learn that function fairly well now we'd like to generate with this model right so how do we do that well we
can start at some random point right like let's let's start over here um and you might think well surely I can
just now predict the noise remove that and then I get my output image and that's great except that you've got to
remember now we're starting from a random point in the space of all possible images it just looks like garbled nonsense and the model's trying
to say well what does the noise look like and so you can imagine here for training the further away we get from
our examples the sparse of our training will have been um but also it's not like it's very obvious how we got to this noisy version
right we could have come from this image over here added a bunch of noise we could have come from one over here one
over here and so this model is not going to be able to make a perfect prediction at best it might say well
somewhere in that direction right it could Point towards something like the data set mean or at least the edge that's closer but it's not going to be
able to perfectly give you one nice solution and sure enough that's what we see if we sample diffusion models just
in one step we get the predictions look at what that corresponds to as an image it's just going to look like a blurry mess maybe like the mean of the data or
you know some sort of garbled output definitely not going to look like a nice image so how do we do better and the idea of
sampling is to say well there's a couple of framings so I'll start with the existing framing that you'll see talked
about a lot of score based models and so on and then we'll talk about some other ways to think about it as well
so this process of gradually corrupting our images away adding a little bit of
noise at a time people like to talk of this as a stochastic differential equation stochastic because there's some
Randomness right we're picking random amounts of noise random directions to add and a differential equation because
it's not talking about anything absolute just how we should change this from moment to moment to get more and more
corrupted right so that's why it's a differential equation and with that framing the question of
well how do I go now back to the image that's framed as solving an ordinary differential equation that corresponds
to like the reverse of this process now you can't solve Odes in a single step but you can find
an approximate solution um and the more sort of sub steps you take the better your approximation and
so that's what these samples are doing given like okay we said this image over here here's my prediction rather than moving the whole way there in one go
we'll remove some of that noise right do a little updates and then we'll get a new prediction
right and so maybe now the prediction is slightly better it says up here so we move a little bit in that direction and
now it makes an even better prediction because as we get closer to the manifold right as we have less and less noise and more and more of like some image
emerging the model is able to get more more accurate predictions and so in some sort of number of steps we divide up
this this process and we get closer and closer and closer until we ideally find some image that looks very plausible as
our output and so that's what we're doing here with a lot of these samples they're effectively trying to solve this ode
in some number of steps by um yeah breaking the process up and only moving a small amount at a time
now you get sort of first order solvers right where all we're doing is just linearly moving within each one and this
is equivalent to something called Euler's method or Euler's method if you're like me and you've only ever read it and this is what some of the most
basic Samplers are doing just linear approximations for each of these little steps um but you also get additional
approaches so for example um maybe if we were to make a prediction from here it might look like something
like this and if we were to make a prediction from here it might look like something like that so we have our error
here but as we move in that direction it's also changing right so there's like a derivative of a derivative a gradient
of a gradient and that's where this kind of second order solver comes in and says well if I know how this prediction
changes as I move in this direction like what is the derivative of it then I can kind of account for that curvature when
I make my update step and maybe know that it's going to curve a bit in that direction and so that's where we get
things like these so-called second order solvers and higher order solvers the upside of this is that we can get you
know do a larger step at a time because we have a more accurate prediction we're not just doing a first order linear
approximation we have this kind of curvature taken into account the downside is that to estimate that curvature for a given point we might
need to call our model multiple times to get multiple estimates and so that takes time so we can take a larger step but we
need more model evaluations per step a kind of hybrid approach is to say well rather than trying to estimate the
curvature here I might just take a linear step look at the next prediction but I'll keep a history of my previous
steps and so then over here it predicts like this so I have now this history and I'm going to use that to better guess
what this trajectory is so I might keep a history of the past you know three or four or five predictions and know that
since they're quite close to each other maybe that tells me some information about the curvature here and I can use that to again take larger steps and so
that's what we see the circle you know linear multi-step sampling coming in just keeping this buffer of past predictions to try and do a better job
estimating than the simple you know one step linear type first order solvers
okay so that's the school-based um sampling version and all of the variants and Innovation comes down to
things like how can we do this in as few steps as possible you know maybe we have a schedule that says we take larger steps at first and then gradually
smaller steps as we get closer um you know there's I think now some Dynamic methods and can we estimate how
many steps we need to take and so on um so that's all trying to attack it from this kind of
score based ode solving framework but there's another way to think of this as well and that's to say okay well I don't
really care about solving this exact reverse ode right all I care about is that I end up with an image that's on
this manifold like a plausible looking image and so I have a model that estimates how much
noise there is right and if that noise is very small then that means I've got a good image
and if that noise is really large then that means I've got some work to do and so this kind of starts bringing up some
analogies to training neural networks because the neural networks we have the space of all possible parameters and
we're trying to adjust those parameters not to solve the gradient flow equation right although that that's you know
possible in theory that you might try and do that we don't care about that we just want to find a Minima we want to find a point where our loss is really
good and so when we're training a neural network that's exactly what we do we set up an Optimizer we take some number of
steps trying to reduce some loss and once that loss gets sort of you know levels off right reduce over time levels
off okay cool I guess we found a good neural network um and so we can apply that same kind of thinking here to say all right I'll
start at some point and I have an estimate of the gradient right it's like maybe pointing over here
but remember that estimate is not very good just like the first gradients estimated when training in your network
are pretty bad because it's all just these randomly initialized weights but hopefully at least points in a useful Direction
so then I'll take some step and the length of this step I won't try and do some fancy schedule I'll just offload
this to a sort of off-the-shelf Optimizer right so I have some learning rate maybe something like momentum that
determines how big of a step I take and then I update my prediction right take another step in that direction and so on
so now instead of following a fixed schedule we can use tricks that have been developed for training neural
networks right adaptive learning rates momentum weight Decay and so on and we can apply them back to this kind of
sampling case and so it turns out this works okay I've tried this for stable diffusion needs some tricks to get it
working but it's a slightly different way of thinking about sampling rather than relying on sort of a hard-coded ode
solver that you've figured out yourself just saying why don't we treat this like an optimization problem where if the
model predicts almost no noise that's good we're doing a good job and if the model predicts lots of noise then we can use that as a gradient and take a
gradient update step according to our Optimizer and try and sort of converge on a good image as our output and this
this is you know you can stop early once your model prediction is sufficiently low for the amount of Noise Okay cool
I'm done and so I found you know in 10 15 Steps you can get some pretty good images out
um yeah so that's a different way of viewing it not so popular at the moment but maybe hopefully something we'll see
um yeah just a different Framing and for me at least that helps me think about what we're actually doing with these Samplers we're trying to find a point
where the model predicts very little noise so starting from a bad prediction moving towards it getting better by
looking at this estimated amount of noise as our sort of gradient and solving that um just kind of iteratively removing
bits at a time so I hope that helps elucidate the different kinds of Samplers and the goal
of that whole thing and also illustrate at least why we don't just do this in a single step right why we need some sort of iterative approach otherwise we'd end
up with just very bad blurry predictions all right I hope that helps now we're going to head back to the notebook to talk about our final trick of guidance
Additional guidance
okay the final part of this notebook guidance how do we add some extra control to this generation process right
so we already have control via the text and we've seen how we can modify those embeddings we have some control by
starting at a noisy version of an input image rather than Pure Noise to kind of control the structure but what if
there's something else what if we'd like a particular style or to enforce that the model looks like some input image or
maybe sticks to some color palette um it would be nice to have some way to add this additional control
and so the way we do this is to look at um some loss function on the decoded
denoised predicted image right the predicted denoise meets final output and
use that loss to then update the noisy latents as we generate in a direction
that tries to reduce that loss so for demo we're going to make a very simple loss function I would like the image to
be quite blue and to enforce that my error is going to be the difference between the blue channel right red green
blue blue is the third channel of the color channels and the difference between the blue Channel and 0.9 so the
closer all the blue values are to 0.9 the lower my error will be so that's going to be my kind of guidance loss and
then during sampling here what I'm going to do everything is going to be the same as before but
um every few iterations you could do it every iteration but that's a little slow so here every five iterations I'm going
to set requires grad equals true on the latents I'm then going to compute my predicted denoised version I'm going to
decode that into image space and then I'm going to calculate my loss using my special blue loss and scale it with some
scaling Factor then I'm going to use torch to find the gradient of this loss with respect to
those latents those noisy latents and I'm going to modify them right and I want to reduce the loss I'm going to
subtract here this gradient multiplied by Sigma squared because we're going to be working at different noise levels and
so if we run this we should see hopefully it's going to do that same sort of sampling process as before but
we also are occasionally modifying our latents by looking at the gradient of the loss with respect to those latents
and updating them in a direction that reduces that loss and sure enough we get a very nice blue picture out and if I
change the scale here down to something lower and run it we'll see that
um scale is lower so the loss is lower so our modifications to the latents are smaller and we'll see that we get out a
much less blue image there we go so that's the default image very red and dark because it's the prompt is just a
picture of a campfire but as soon as we add our um our additional loss or guidance we're
going to get out something that better matches that additional constraint that we've imposed right so this is very
useful not just for making your images blue but like I said color palettes or using some classifier model to make it
look like a specific class of image or using a model like clip to again associate it with some text so lots and
lots of different things you can do now a few things I should note one we decoding the image back to image space
calculating a loss and then tracing back that's very computational Lee intensive compared to just working in latent space
and so we can do that on every fifth operation to reduce the time but it still is much slower than just your
generic sampling and then also um we're actually still cheating a little bit here because what we should
do is set requires grad equals true on the latents and then use those to make our noise prediction use that to
calculate the denoise version and decode that calculator loss and Trace back all
the way through the decoder and the process and the unit back to the latents right the reason I'm not doing that is
because that takes a lot of memory so you'll see for example like the clip guided diffusion notebook from the
hugging face examples they do it that way but they have to use tricks like gradient checkpointing and so on to kind of keep the RAM usage under control and
for simple losses it works fine to do it this way because now we're just tracing back through denoise latents is equal to
latents minus Sigma times this noise prediction right so we don't have to trace any gradients back through the unit but if you wanted to get more
accurate gradients maybe it's not working as well as you'd hoped you can do it that other way that I described
um but however you do it very very powerful technique fun to be able to again inject some additional control into this generation process by crafting
a loss that expresses exactly what you'd like to see all right that's the end of The Notebook
for now if you have any questions feel free to reach out to me I'll be in the forums and you can find me on Twitter and so on but for now enjoy and I can't
wait to see what you make

---

# 9B

hello everyone uh my name is Waseem I am an entrepreneur in Residence at fast AI
and um I'm currently at the first AI headquarters at the moment in in
Australia although I'm originally from South Africa from Cape Town
um and I'm joined here today by tanishq tanishq works at stability Ai and we've
been working together with a couple other people on diffusion models generative kind of modeling
and that's been super fun so tanishq do you want to maybe you know introduce yourself as well
yeah so my name is tanishq I am a PhD student at UC Davis but I also work at
stability Ai and I've been exploring and playing around with diffusion models for
the past several months and so it's been it's been great to also uh explore that
with the with the fast AI Community as well in these last few weeks as well awesome
cool cool so this talk is um
us or me trying to understand the math behind diffusion so you know if you've done the fast AI
courses before you know that you don't need to understand the math to be effective with any of these models in
fact you don't even need the math to do research to do novel research and contribute to these
but for me it was all about it came out of interest and you know I thought it
was kind of it's kind of beautiful how uh what you know diffusion models were
discovered and I think a large part of that was thanks to some some really clever mess
and so I wanted to understand that um I'm not I don't have a math
background um and so I
want to help kind of describe how I think about it and how I um how
you can kind of interpret you know all of these notations and things
cool um yeah so so I can just dive into it I think
Data distribution
so the first sort of math that we see in this paper
is Q of x superscript 0. and they call this the data distribution
do you want to mention what the paper exactly which paper this is right good good question so this paper
is the 2015 paper uh do you remember the authors that I paid for tanishq uh I
think it's just a soul dickstein Yahoo networks at Google I think and it's from
uh Surya ganguly's lab so cool yeah yeah so so this was the papers as far as I
understand that introduced this idea of diffusion um yeah 2015 by those authors
they start out by you know defining this data distribution and they use this
notation and already uh like a lot of people you know myself included Finance quite confusing
um but let's go through what's described here so they have an X
and you know in math X is often used as the input variable
um much like Y which is then used often as the output
variable um
yeah and the the fact that it has a superscript also implies something so
the fact that we have X superscript zero implies that there might be a sequence
of x's and you know I think it's it's useful to get comfortable with this idea of
simple compact notations implying a lot more than you know might
be obvious at first glance so X implies that it means something about this quantity it's an input
variable and you know the zero implies that there might be other things so you might have
you might have an X1 an X2 and so on but but we'll we might see that
and then the third part is you have q and Q is a what we call a probability
density function so the first part here is is probability
uh and the question is you know what is what does q have to do with probabilities
well it's because usually we use the letter p to describe probability density
functions of interest and then because Q is right after that it's another common one so it's kind of like how you use X
and Y we use p and Q and the fact that we use Q here instead of B is because it it
suggests that there might be a p that will introduce and maybe p is the thing that we're modeling and Q is kind of
supplementary to that does that sound right tanishq
yeah yeah and I think it's also helpful to kind of maybe think about like x x
zeros in a more practical concrete way of course if we're working with images then x0 would be you know that's that's
what's representing the images so it's also useful to think about it from kind of that concrete practical approach as
well right so x0 might be you know an amnest
digit and then we got Q so Q
I'll just use this to mean uh Q is some function so we look at it
as a box
and it takes in X zero and it gives us uh the probability
that this x 0 which is an image looks like an amnest digit so in this
case you know this would be 3.9 or maybe even uh yeah let's go
0.98 so this is quite high probability that this is the name this digit hi this
Math behind lesson 9’s “Magic API”
is Jeremy can I jump in for a moment please do oh thank you I just wanted to double check
um this looks a lot like the magic uh API that we had at the start of lesson one that you feed in a digit and it
gives you back a probability is that basically what what Q is doing here absolutely yeah it's a magic API it's a
good way to think of it we don't know we couldn't write down what Q is but we
imagine that somebody somebody has it somewhere um yeah so so this is a concrete example
and like if you had to do something to this image uh you might get a smaller number
so another thing worth mentioning here is probability density functions so these
are these magic apis that you know give us the number tells us How likely the thing is
they you don't often see them they don't often make it all the way to your code in fact they very rarely will appear in
your code um but turns out that there are very useful ways or tools to work with random
quantities because they allow you to represent random quantities as functions just
ordinary functions and because they're functions you have a whole you know centuries worth of math
to analyze and understanding so you'll often find probability density
functions in papers and eventually they work out to really simple equations or
formulas that end up in your code
do you want to add anything to Niche I think that sounds so correct of course
um I think you probably will go over some examples of probability density
functions especially relevant to this one but yeah it's useful to think about the process that the sorts of functions
you may have in a in a simplified case and that's what we probably are going to talk about next right yeah yeah that's
exactly what we're talking so we have this qx0 and then we introduce another
one and like you said this is going to turn out to have a really nice simple form
but before that the next thing we Define is qxt
minus one so we will say what we Define this to be
but to begin with this is another probability density function
and this bar over here means it's a conditional probability density function
which you can think of as you are given the thing on the right to
calculate uh probabilities over the thing on the left in this case
it you can think of it as something that takes images
so maybe another magic API
and produces other images but we don't know what these look like
yet because we haven't defined over here and this this we would call kind of you know x t minus one which
could be x0 and this would be x t which in the x0
case would be X1 something worth noting here is
this notation can be a little bit confusing because we we said Q is one
thing earlier now we think Q is another thing so this year I'm going to need your help
on the song tanishq um I think people would usually if in the
stricter sense Define the first one uh you know
like this maybe and the second one
was a subscript
and that this notation that we see here on the left
is just a shortcut where they you know they wanted to save the space of writing
that and kind of included that you were implied it by what was in the practice is that
true yeah I mean I think here they use the the
first later on vcp to kind of describe as we'll see different aspects of the
diffusion model uh the sort of different processes of the diffusion model which
we'll see so I think that's what you know those they use the same variables to kind of demonstrate this is
corresponding to this process and then yeah the variable corresponds to the other process of the diffusion model so
we'll obviously go over that so I think that's where that those those variables
or those uh letters are being used in that manner but if you do want to make it for more specific more clear yeah I
think that that that notation is fine as well all right okay yeah that makes
sense okay so so let's describe what this
queue does uh you know to the image on the left to produce the one on the right so I'll start over here so we have more
space
I'll write it out first and then we can go into the details
foreign so kind of like the bar you can think of
this semicolon as you know grouping things together
and so you have the things on the left and things on the right my understanding is these two things on
the right are the parameters
of the model uh sorry of the probability the thing on the left is
actually Denise could you help me understand what the thing on the left is do you know right well so this is again
like a probability distribution and the thing on the left is saying this is a probability distribution for this
particular variable so that's just representing what it is a probability distribution for and then the stuff on
the right are the parameters for this uh probability uh distribution so that's
kind of what's going on here so like yeah anytime you have like a normal distribution and it's describing some
variable you'll have that sort of notation right it's the normal distribution of some variable
um and then these are the parameters that this would not describe that normal distribution
right
so just to clarify the bit after the semicolon is the bit that we're kind of used to seeing to describe a normal
distribution which is which is the mean and variance of the normal distribution
so we're going to be sampling random numbers
from that normal distribution according to that mean and that variance is that right
yes that's correct yeah mm-hmm
yeah so we we need to describe a bit more there about normal distribution we kind
of you know Skip past that so we have this Fancy in and fancy letters in math for
distributions usually refer to well-known distributions
and the N here stands for normal which is also known as
a gaussian distribution [Music] and it's probably the most well-known
probability distribution that you can you can find and what when I say well known I mean
that these things pop up everywhere um you know you can do in all sorts of
fields measuring all sorts of things turns out that they follow roughly
something that looks like this distribution and because they pop up so much
um you know people studied them studied all of their properties and we
understand them really well now the reason that they used often in cases
like this is because they turns out they have really useful properties and they're easy to work with
some reasons are they are described by just two parameters
so the mean called the mean and the covariance another property is that they have kind
of you know what people would call Sun tails which kind of means that they only you
only need to describe their behavior in a small region of space uh you can kind of just ignore the rest
um yeah do you mind drawing a quick example of a normal distribution that's a good
point so we have let's say our random variable is just one kind of dimensional
so just a single number of floats this is sort of what the normal
distribution would look like and in this case that would be our mean
and the variance would sort of describe the what's over here
which in this case you'd use a small Sigma because you're doing a single variable
in our case we use a capital Sigma which is the symbol for multiple variables or
multiple dimensions and yeah I also didn't say that this is
the letter Greek letter mu so it's capital Sigma mu and lowercase Sigma
uh I just wanted to note that typically the lowercase thinking about represents
the standard deviation which is the square root of the of the of the variance
um so for example sometimes you may see uh in papers uh Sigma squared and that's
just the the variance but they will write it sometimes as Sigma squared instead
the sigma is a standard deviation often and sigma squared would then be the
variance cool uh yeah we can also show with our
example what what this would look like so we start out with
a MNS digit
put it through this magic API and what would we get out
okay so oh something we didn't describe is you know what does this I mean
did you want me to talk about that was him ah yes please okay sure so because I
think this is something which actually can I borrow your pen it actually came up in
um the lesson we were doing kind of in an interesting way so um
oh well okay I'm in the video now um yeah in the video uh hi Tanish nice
to see you hello yeah so in the lesson like we we did this thing for clip I
CLIP (Contrastive Language–Image Pre-training)
don't know if you remember um Russian where we had the um you know the various pictures down
here I'm so embarrassed you're better at the graphics tablet than I am and it's my Graphics tablet and we had the various
sentences along here right and we said oh you know it'd be kind of cool to like take the dot product of their embeddings
because like if their dot products are high that means they're they're similar to each other
um and you know if if we subtracted the means from those first right then you've
got the the dot product and instead of having um images down here okay what if we had
the exact same um the exact same uh vectors on each side
then what you've got down here is basically x minus
you know it's average right if you subtract that first squared
and that is the variance right so that's like the variance for each one of these vectors
but what's interesting as you pointed out is that like normally you know at high school when we look at
a normal distribution it looks like this right but you're not just doing one normal distribution you've got a whole
bunch of kind of normal distributions right for all of your different uh pixels they're the pixels right tanishq
normally sort of the distribution of every pixel so there's a whole bunch of them and so one of them might have a normal
distribution that's there and another one might have a normal distribution that's here and another one might have a normal distribution that's like here
and it's more than that though because like it's possible that that you know
two one pixel tends to be higher when another pixel tends to be higher or one pixel tends to be higher when another
pixel is lower so it actually has kind of kind of create this like surface
you know an n-dimensional space to add as the number of pixels
so if you now like look at like okay well what happens if we multiply this by this just like we did in clip right
then if this number is high then it's saying that when this variable is high
where this pixel is high this pixel tends to be high and vice versa or if it's low it's saying when this pixel
tends to be high this one tends to be low or interesting to us what happened oopsie Daisy sorry about that
what happens if this is zero
that says that if this is pi then this could be anything when this is high this could be anything there's no
relationship between them so statistically we would say that these two pixels
are independent and so now that basically means we could
do that for all of these we could say oh you know these are all zeros
and what that says is that oh every pixel is independent of every other pixel now of course in real pictures
that's uh not how real pixels work but that's the Assumption we're making because if we start with a very special
Matrix called I which is one one one one zero zero zero
right if we take this very special matrix it's very special because I can multiply it
by something say beta and um uh If I multiply it by a matrix I get
back the original Matrix If I multiply it by a scalar I'm going to get beta beta beta and lots of zeros
and so if I multiply something by this Matrix all right then I'm just multiplying it
by Beta but what's interesting about this is that this is what theme wrote
last name wrote I times Theta I times beta t so what he's saying is oh we've now got
a covariance matrix where for each individual pixel
it's like pixel Number One beta one picks number two beta two this is the variances of each one
and the covariances you know the relationship between the pixels is zero
they're expected to be independent so that's where we're kind of going from like
statistics you do in high school to statistics you do at University is like suddenly covariance is now a matrices
not individual numbers does that sound about right to you tanishq yeah that's that's a that's a great explanation of
it yes awesome
cool so now let's let's try to describe you know what this would do with two Ms
digits so you know we let's put back our mean
equation
and our covariance whoops our covariance
so mean and our covariance and let's look at how this behaves you know at the
edges sort of so it's really hard to you know understand this I don't think anybody can kind of
just look at this and and know what it means what we typically do is we try to
describe it kind of at the edges and so we'll start with like what what happens if that's
zero and we'll work with X here as well instead of you know x t minus one
uh which would mean like an Eminence digit so if if beta zero
and we get our our X zero uh you know square root one minus zero
which is one and square root of one is one so that kind of Falls away so we just have
a mean of our previous image and this is just variance of zero
so we have a normal distribution with the mean of our previous image a variance of zero
which means we have the same image
yeah just to clarify when you have variance of zero that means that there's really no noise or anything it's just at
that mean and you know your distribution is just saying that's the only point that you can get from it so yeah that's
what it just becomes the same image because uh yeah there's no noise or variance because of the variance is zero
yeah exactly and then when our beta is one
we still have this and then we have you know square root one minus one
and that becomes zero so this whole thing becomes zero
and this thing becomes I times Theta T which is you know I
and if it's just I then as Jeremy described it would you know imply a variance of one
and so our image through this function
would just be Pure Noise so let you know mean of zero
standard deviation of one and it would just be a bunch of noise
and kind of somewhere in between that we have to say over here you know what
would it produce it would be some mixture so you know
like maybe a light the lighter pixels of eight and some noise
maybe a bit darker
Forward diffusion (markov process with gaussian transitions)
and we can kind of draw this and and you you would have seen this in the previous lecture
you can draw the sequence
of things that become progressively more noisy in very small
steps all the way until it becomes Pure Noise this is what we call the forward
diffusion process
and we can now describe some of these things so this would be a sample from
our data distribution Q x0
this would be the function for the conditional
probability density function that takes so of X1
given x 0 and so on
and the way that the the terminology that we
would use or that mathematicians use to describe this as they would call it a mark of
process with gaussian transitions
and you know this this can sound quite scary but we've just described exactly what this is
so when we say process it usually means you know something where there's a sequence involved
when we say Markov it means that the thing
at time T depends only on the thing at T minus one
the transition is this function how do you actually go from T minus one
to T and gaussian is the fact that that transition is the normal distribution
does that sound right yes uh just to also clarify a couple
things and we say that you know we're sampling from the data distribution what
that is referring to is trying to find some random
You Know sample or some random data point that maximizes that likelihood or
that has a high likelihood so when we say that you know we're looking at that that API that magic apis we were talking
about and we're trying to get some you know uh some data points that have a high value with you know from that API
and you know for some so for sub distributions but it's very simple and they know how it works like a
gaussian distribution if you know the parameters of that gaussian distribution it's very easy to be able to do that
sampling and then of course in other cases it's not very it's not it's quite difficult to do that sampling so then we
have to figure out alternative ways of doing that sampling but that's why in this case with the forward distribution
we just have these simple gaussian Transitions and we already know the parameters of those gaussian transitions
so we can easily do that sampling and going back also to that I think it's a
worthwhile to also kind of show and think about maybe how this is again done practically because one of the nice
properties of gaussian distributions as a whole is that you can uh you know
simply take some normal noise that with a mean of zero and variance of one so
that's I think they usually typically call that a unit uh uh unit distribution
it's just like yeah normal of zero one and then if you want to get to some
other point with a mean of whatever value you specify and a variance of
whatever value you specify you can simply uh take that normal distribution
scale it by the um you multiply it by the variance and then
you add your your mean so then there's a simple equation that you can take to to
get the uh to you know to get at any particular uh mean and variance so
that's how you would you know for either get the samples for these other uh
distributions that we have defined uh throughout the forward distribution so you know for example when you're coding
this up of course A lot of these uh softwares they will have a way of
getting a sample from this normal distribution of zero one and then you
just use that equation then to get it at the desired mean and variance and so that's how it kind of happens uh under
the hood when you're when you're kind of described this uh with code
that's really helpful yeah and this idea of we can't really
sample from this thing um that's exactly you know the problem that generative kind of modeling is is
trying to solve like how do you represent this in such a way that you can easily sample from it
and so it turns out that if you have one of these persisters
you know where you have many many steps so let's say a thousand steps a thousand of these tips going to the right
and they're all very small steps that eventually go to noise
somebody uh you know maybe in the 1950s
I think discovered that you can represent the process of going
backwards in exactly the same functional form
with just different parameters so what that means is if we say p is the thing that goes back with so
you know the previous one given the current one
this p has the same functional form so
it's also the transitions are also normal but the mean is you know some unknown so
we'll use a square and the variance is some unknown so we use a triangle
um is that correct yeah that's correct and
just going back to our previous point about P versus Q here we can see that that the queue was describing the sort
of forward process going you know yeah this sort of steps that we're doing and then the p is describing when we're
going in the reverse uh uh reverse way so that's why you know these papers are
using you know Q for some one one process in the P for another that's what they're kind of indicating at least in
the diffusion model literature and peas is kind of like X you know it's
the it's the one we want to figure out so like Q is kind of like Y and the p is
kind of like exits how I like to think of that and so you you know we have this functional form
and the next question is uh how can we use this so you know we we
just don't know what these parameters are how can we figure out what those are
and this is uh goes back you know to early kind of Statistics literature
where you can fit this model using by
maximizing What's called the likelihood function so we can try different parameters
until we have one that maximizes the likelihood
it turns out that we can't quite do this exactly
because you would need to calculate some integral and that integral is over very high
dimensional values continuous values so you can't actually calculate this
I I think you can think of it because you know we're having these uh thousands
of steps that we're trying to go in this reverse process and so you know you have these thousands of steps that they're
going to be many possible values for each step so it's kind of hard to evaluate it over all these thousands of
steps and all the possible values for all these different steps so I think that's kind of where the the challenges
arise and that's what it makes it difficult because you have to find uh you have to evaluate it over these
multiple steps and try to find these functions for all these different steps so that's that's kind of where the
challenge is mm-hmm
Likelihood vs log likelihood
and so you might see people talk not about the likelihood function but about the log likelihood
and correct me if I'm wrong here tanishq but I think the log here is is a bit of
a you know computational tick almost so I think it has a few properties the first is
that it's it's always increasing and you know people would call this I think monotonic
uh you know it looks always kind of increasing and because it's always
increasing if it's the same you get the same parameters if you optimize the log
likelihood versus you optimize the likelihood it also uh takes uh products to sums
because um and and that's helpful because we have joint distributions you know which
turn out to be products so it turns out we have a lot of products here and they become songs which is easy to work with
and the last thing is that uh you know this normal distribution has
exponentials or exponential functions and those uh disappear with the log
so this is a much friendlier thing to optimize yep that's correct
cool and then there's one more step uh you know we we still can't optimize the log
likelihood of the thing that this eventually describes but again and this is kind of the beauty
of math is that somebody figured out a long time ago that there's a way to optimize some other quantity
uh called the elbow for short
which stands for evidence lower bound
and the evidence is just another name for the likelihood
and the lower bound means uh it's sort of you know it's the lower bound of the evidence then if you
optimize that it's almost as good as optimizing the thing that we really want to but this one we can calculate very
very easily and so you can use this
as a loss function
to train two neural networks
that predict our Square from earlier which was our mean
and our triangle which is our variance of this reverse process
and once you have that you can go all the way back here so then you have these values
you can start with Pure Noise and keep calling these neural networks
sampling from those normal distributions
um kind of applying that iteratively over many steps and you recover this
data distributions one thing that's important to clarify
here is that you can recover the whole distribution but you can't necessarily take a single
image convert it to Pure Noise and then convert it back
so this operates sort of at the distribution level so you can take this kind of magic API you
can reconstruct that whole API and if you can do that then you know you
can generate images digits or cats or dogs or whatever you want to
I want to just clarify one thing about this process of the kind of the the loss
function so this sort of evidence lower bound loss function the kind of approach
that it's taking is that you know we have this forward process right we have we can go from the original images and
you know figure out these sorts of intermediate distributions going all the way
finally to noise with this sort of evidence level bound um loss function what we're really kind
of doing is trying to match the power distribution that we're trying to
optimize to those distributions that we saw in the forward process so that's what we're trying to do we're trying to
match uh that sort of uh those distributions and there's a specific
type of of uh function that is able to do that it's called a KL Divergence
that's the sort of function that can compare probability distributions and
again because we're dealing with gaussians uh you can calculate that analytically and a lot of the math uh
you know becomes very simple so that's again you know with the whole gaussians you know we we know them quite well the
math is very simple so that allows us to do this sort of comparison between these distributions very very easily and
optimize that and so we want to kind of minimize the difference between the distributions we see in the forward
process and the distributions we're trying to determine for the reverse process
perfect then there's there's one more thing I think one more kind of major you know
step to get closer to the form that that you would have seen in Jeremy's lesson
Denoising diffusion probabilistic model (DDPM)
um so there was the 2020 paper uh the initials of that model is ddpm
tanishq you know what this stands for yeah it's that's for denoising diffusion
probabilistic model okay cool
and what they did was they said let's assume that this variance is just
a constant so we don't learn it
and we assume also that the step size from earlier you know the variance of
the noise that we add at each step is also a constant we don't learn that
so we're just predicting the mean and these are set to some really convenient values
then the loss turns out to be
that you predict the noise
so you can restructure this whole thing as you take in you need to train a network
that takes in images so here's your network
and it tells you what of this image is noise
thanks to these you know these simplifying assumptions and even though the assumptions
turns out you can train much more uh you know models that produce much better
images now I think this relates to something from
the you know the lesson that that Jeremy gave tanishq do you remember that there was
something about the gradient or something like that yes yes so
um this idea of you know adding noise and and learning to remove noise uh the
idea is that kind of by uh you know again you have this sort of uh
this image that you have noise right and by
sorry let me uh think about the best way to say this
uh oh yeah sorry okay let me start it over so I'll just start
um yeah so like uh Jeremy will say uh in his uh in the lesson what we want to do
is we want to figure out the the gradient of of this likelihood function
so this is just kind of a different way about thinking about this if we had some information about this gradient then we
could for example um you know use that information to
uh produce kind of like we talked about kind of this optimization kind of produce images with high likelihood so
the idea is that we can add noise to the to the images that we
have so that's those are samples that we have and that kind of uh takes us away
from you know the regular images that we know that we have and you know that kind of decreases the likelihood right so we
have those images and we're adding noise that decreases the whole likelihood and we want to kind of learn how to get back
to high likelihood images and and kind of use that to provide some sort of
estimate of our gradient so this sort of denoising process actually allows us to
do that so there are actually uh theorems also I think from the 1950s
that demonstrate that especially in the case of this sort of gaussian noise that we're working with uh this denoising
process is equivalent to learning uh what is known as the the score function
and the score function is the gradient of the log of the likelihood so again
they have this log here which again make makes the math nicer and easier to work with but the general idea is the same
because as as we talked about log is a monotonic function so again the general
ideas are the same but the score function specifically refers to the
gradient of the log likelihood so this sort of denoising process allows us to
learn the score function so that's what we're doing this noise predicting that
you know we've had this whole probabilistic framework using that sort of likelihood framework and it came back
down to just predicting the noise and that's what the ddpm paper showed in 2020 but it turns out that is equivalent
to calculating out this sort of score function and using that information to
be able to uh sample from our distribution so that's kind of how these
two approaches connect so there's a lot of literature talking about maybe the that sort of holistic likelihood
perspectives of diffusion models and there's also a lot of literature talking about this score based perspective but
you know this hopefully allows you to think about the similarities and how these two approaches connect with each
other yeah awesome yeah and that's kind of the you know the
beauty I think of the math side of things here is that you find all of these relationships
um between different fields and also like between different centuries basically and that allows you to do
really kind of powerful and unexpected things Okay so
Conclusion
you can just do a quick recap where we we got to so we started out
with our data distribution which we want to model uh we said you know we'll Define this
forward diffusion process which is a way of kind of adding noise to this model
and because we added in this specific way thanks to you know some Discovery in the
1950s uh the reverse process has the same form
and then you know we already know how to train a neural network for this
using the elbow
and then a couple years later came the discovery uh you know simplifying assumptions that in the end all we do is
predict the noise and I just remembered we take actually the MSE of this noise
prediction the mean squared area which is a nice very simple framing of the model and internist spoke about the
another way to derive all of this which is the score function approach the gradient of the log
likelihood okay cool um yeah I highly recommend
checking out the course lesson as well if you haven't um
you know if you don't understand this there's no need to be intimidated
uh you can still do be very effective without ever using math you can be very
effective at Deep learning as fast AI has shown us and you can do novel research as well
for me this is it's interesting and um you know it's even beautiful in a way so
I I recommend checking it out but don't feel intimidated you can find the course
lesson links in the past AI Forum We'll add those links as well in the
description of this video we'll also have a topic in the forum for this lesson you can have discussions
there post any comments add any you know relevant links to the math
and then we have another lesson uh you know video by jono
which I I really recommend checking out he's a you know he's a great teacher and he was I think he was the first person
to do a full course on on stable diffusion yeah jono's video is kind of a deep dive into some of the code a little
bit more and into some of the concepts a little bit more so I feel like between these three videos it's a
a good overview you know I think uh I mean just to clarify
you don't need to understand all the math that was being described in this video that's not to say you want me to
understand math we'll be covering lots of math um in these lessons
um but we'll be covering just the math you need to understand and build on the code
and we'll be covering it over many many more hours than this rather rapid
overview perfect cool and yeah thank you so much
Danish I had a lot of fun and thank you so much question that was awesome
cool bye-bye

---

# 10

Hi everybody and welcome back. This is lesson  10 of “Practical Deep Learning for Coders”.  
It's the second lesson in Part 2, which is where  we're going from “Deep Learning Foundations to  
Stable Diffusion”. So before we dive back  into our notebook, I think, first of all,   let's take a look at some of the interesting  work that students in the course have done  
over the last week. I'm just going to show  a small sample of what's on the forum so   check out the “Share your work here” thread  on the forum for many, many more examples.
Showing student’s work over the past week.
So @puru did some, something interesting, which  is to create a bunch of images, of doing a linear  
interpolation —I mean, detail is actually  spherical linear interpolation but it doesn't   matter—. Doing a linear interpolation between  two different latent, you know, noisy, you know,  
latent noise starting points for an otter picture  and then showed all the intermediate results.  
That came out pretty nice and then did something  similar starting with an old car prompt and going  
to a modern Ferrari prompt —I can't remember  exactly what the prompts were— but you can   see how, as it kind of goes through that latent  space, it actually is changing the image that's  
coming out, I think that's really cool. And then  I love the way @namrata took that and took it to  
another level in a way which is starting with  a dinosaur and turning into a bird and this is  
a very cool intermediate picture of one of the  steps along the way: the dino-bird . I love it.  
Dino-chick, fantastic. So much creativity on  the forum, so I loved this, John Richmond took  
his daughter's dog and turned it gradually into  a unicorn and I thought this one along the way  
actually came out very, very nicely. I think this  is adorable and I suspect that John has won the  
dad of the year or dad of the week, maybe,  award this week for this fantastic project.
And Maureen did something very interesting, which  is, she took Johno's parrot image from his lesson  
and tried bringing it across to  various different painters styles  
and so her question was: “anyone want  to guess the artists in the prompts?”.  
So I'm just gonna let you pause it before  I move on, if you want to try to guess…  
And there they are, most of them are pretty  obvious, I guess, I think it's so funny that  
Frida Kahlo appears in all of her paintings so  the parrot actually turned into Frida Kahlo —all  
right, not all of her paintings but all of her  famous ones— so the very idea of a Frida Kahlo  
painting without her and it is so unheard of  that the parrot turned into Frida Kahlo. And   I like this Jackson Pollock, it's still got the  parrot going on there. So that's, that's a really  
lovely one Maureen, thank you. And this is a good  reminder to make sure that you check out the,  
the other two lesson videos so she was  working with Johno's Stable Diffusion  
lessons, so be sure to check that out if you  haven't yet, it is available on the course webpage  
and on the forums and has lots of cool stuff  that you can work with, including this parrot.
And then the other one to remind you about is  the video that Wasim and Tanishq did on the math  
of diffusion and I do want to read out what Alex  said about this because I'm sure a number of you  
feel the same way: “My first reaction on seeing  something with the title ‘the math of diffusion’   was to assume that ‘oh, that's just something  for all the smart people who have PhDs in maths  
on the course, and it'll probably be completely  incomprehensible’, but of course it's not that   at all!”. So be sure to check this out even if you  don't think of yourself as a math person, I think  
it's, you know, some nice background that you may  find useful —it's certainly not necessary— but you  
might, yeah, I think it's kind of useful to start  to dig in some, to some of the math at this point.
One particularly interesting project that's been  happening during the week is from Jason Antic, who  
is a bit of a legend around here, many  of you remember him as being the guy that   created DeOldify. And actually worked  closely with us on our research which,  
together, turned into NoGAN and Decrapify  and other things, created lots of papers.  
And Jason, yeah, Jason has kindly  joined our middle research team   working on the stuff for these lessons and  for developing a, kind of, a fastai approach  
to Stable Diffusion and he took the idea that I  prompted last week, which is maybe we should be  
using classic optimizers, rather than differential  equation solvers, and he actually made it work  
incredibly well already within a week. These faces  were generated on a single GPU, in a few hours,  
from scratch by using classic, classic Deep  Learning optimizers, which is like an unheard  
of speed to get this quality of image and we think  that this research direction is looking extremely,  
extremely promising. So really great news there  and thank you Jason for this fantastic progress.
Recap Lesson 9
Yeah, so maybe we'll do a quick  reminder of what we looked at last week.  
So last week I used a bit of a mega, Onenote,  hand-drawn thing. I thought this week I might just  
turn it into some slides that we can use. So the  basic idea, if you remember, is that we started  
with… if we're doing handwritten digits, for  example, we'd start with a number, a number seven,  
this would be one of the ones with a  stroke through it, that some countries use.   And then we add to it some noise, and the  seven plus the noise together would equal this  
noisy seven. And so what we then do is we  present this noisy seven, as an input, to a U-Net  
and we have it try to predict which which pixels  are noise, basically, or predict the noise.  
And so the U-Net tries to predict  the noise from the from the number,   it then compares its prediction to the actual  noise. And it's going to then get a loss, which  
it can use to update the weights in the U-Net and  that's basically how Stable Diffusion the main  
bit, if you like, the U-Net, is created. To make  it kind of easier for the U-Net we can also pass  
in an embedding of the actual digit, the actual  number “7”, so for example one-hot encoded vector,  
which goes through an embedding layer, and the  nice thing about that, to remind you, is that  
if we do this then we also have the benefit that  then later on we can actually generate specific  
digits by saying: I want a number seven or I want  number five and it knows what they look like. I've skipped over here the VAE/latents piece  which we talked about last week. And to remind  
you that's just a computational shortcut, it  makes it, it makes it faster and so we don't  
need to include that in this picture because  it's just a, yeah, it's just a computational   shortcut that we can pre-process things into  that latent space with the VAE first, if we wish.  
So that's what the U-Net does. Now, then,  to remind you, you know, we want to handle   things that are more interesting than just  the number seven, we want to, actually,  
handle things where we can say for example:  “a graceful swan” or “a scene from Hitchcock”  
and the way we do that is we turn these sentences  into embeddings as well, and we turn them into  
embeddings by trying to create embeddings of  these sentences, which are as similar as possible   to embeddings of the photos or images that they  are connected with, and to remind you the way we  
did that or the way that was done, originally, as  part of this thing called CLIP, was to basically  
download from the internet lots of examples,  of lots of images, find their ALT tags and then  
for each one we then have their image and its ALT  tag so here's the graceful swan and it's ALT tag,  
and then we build two models: an image encoder,  that turns each image into some feature vector  
and then we have a text encoder, that turns  each piece of text into a bunch of features,  
and then we create a loss function that says that  the features for a graceful swan, the text, should  
be as close as possible to the features for the  picture of a graceful swan, and specifically we 
take the dot product. And then we add up all the  green ones, because these are the ones that we   want to match, and we subtract all the red ones,  because those are ones we don't want to match:  
those are where the text doesn't match the  image. And so that's the Contrastive Loss  
which gives us the CL in CLIP. So that's a review  of some stuff we did last week and so with this  
then we can, we now have a text encoder which we  can now say: “a graceful swan” and it will spit  
out some embeddings and those are the embeddings  that we can feed into our U-Net during training.
And so then, we don't haven't been doing  any of that training ourselves except for  
some fine tuning because it takes a very long  time on a lot of computers. But instead we take  
pre-trained models and do inference, and the  way we do inference is we put in an example of  
the thing that we want, that we have an embedding  for, so let's say, we're doing handwritten digits,   and we put in some random noise into the U-Net  and then it spits out a prediction of which  
bits of noise you could remove to leave behind a  picture of the number three. Initially it's going  
to do quite a bad job of that, so we subtract  just a little bit of that noise from the image  
to make it a little bit less noisy, and we  do it again, and we do it a bunch of times.  
So here's what that looks like, creating a —I  think somebody here did “a smiling picture of  
Jeremy Howard” or something, if I remember  correctly— and if we print out the noise at,  
kind of, step 0, and at step 6, and at step 12,  you can see this first signs of a face starting to  
appear. Definitely a face appearing here: 18, 24.  By step 30 it's looking much more like a face. By  
42 it's getting there, it's just got a few little  blemishes to fix up, and here we are. I think I've  
slightly messed up my indexes here because it  should finish at 60, not 54, but such is life.  
It's a rather rosy red lips  too —I would have to say—.
So remember, in the early days this took  a thousand steps and now there are some  
shortcuts to make it take 60 steps. And this  is what the process looks like and the reason  
this doesn't look like normal noise is because  now we are actually doing the VAE latents thing  
and so, noisy latents don't look like  gaussian noise, they look like, well,  
they look like this, this is what happens  when you decode those noisy latents.
Now, you might remember, last week I complained  that things are moving too quickly and there was a  
couple of papers that had come out the day before  and made everything entirely out of date so,  
Johno and I and the team have actually now had  time to read, read those papers and I thought  
Explaining “Progressive Distillation for Fast Sampling of Diffusion Models” & “On Distillation of Guided Diffusion Models”
now would be a good time to start going  through some papers for the first time. So the…  
what we're actually going to do is show how these  papers have taken the requirement number, required  
number of steps to go through this process down  from 60 steps to 4 steps, which is pretty amazing.
So let's talk about that and  the paper is, specifically  
is this one “Progressive Distillation  for Fast Sampling of Diffusion Models”.  
So it's only been a week so I haven't had much  of a chance to try to explain this before so   apologies in advance if this is awkward but  hopefully it's going to make some sense.  
What we're going to start with is, so  we're going to start with this process  
which is gradually denoising images —and  actually I wonder if we can copy it.  
Okay, so, how are we going to get this down  from 60 steps to 4 steps? The basic idea  
is that we're going to do a process. We're  going to do a process called “Distillation”  
—which I have no idea how to spell but hopefully  that's close enough that you get the idea—.   Distillation is a process which is pretty  common in Deep Learning and the basic idea of  
distillation is that you take something called a  Teacher Network, which is some Neural Network that  
already knows how to do something, but it might be  slow and big. And the Teacher Network is then used  
by a Student Network which tries to learn how to  do the same thing but faster or with less memory.  
And in this case we want ours to be faster.  We want to do less steps. And the way we  
can do this conceptually, it's actually, in my  opinion, reasonably straightforward. We have,  
like, when I look at this, and I think like:  “wow, you know, Neural Nets are really amazing”.  
So, given Neural Nets are really amazing, why  is it taking, like, 18 steps to go from there  
to there. Like, that seems like something  that you should be able to do in one step.  
The fact that it's taking 18 steps —and  originally, of course, that was hundreds and  
hundreds of steps— is because it's kind of that's…  that’s just a kind of a side effect of the, of the  
math of how this thing was originally developed,  you know, this idea of this diffusion process.
But the idea in this paper is something  that actually we've —I think I might have  
even have mentioned in the last lesson, it's  something we were thinking of doing ourselves   before this paper beat us to it—. Which is  to say, well, what if we train a new model  
where the model takes as input this image, right?  And puts it through some other U-Net… U-Net  
B, okay? And then that spits out some result.  
And what we do is we take that  result and we compare it to  
this image —the thing we actually  want—. Because the nice thing is now,  
which we've never really had before is, we  have, for each intermediate output, like,   the desired goal where we're trying to get to.  And so we could compare those two, just using,  
you know, whatever, Mean Squared Error. Keep on  forgetting to change my pen, Mean Squared Error.  
And so then, if we keep doing this for lots  and lots of images and lots of lots of pairs   in exactly this way, this U-Net is going to  hopefully learn to take these incomplete images  
and turn them into complete images. And that  is exactly what this paper does. It just says,  
okay, now that we've got all these examples of  showing what step 36 should turn into at step 54,  
let's just feed those examples into a model.   And that works. And you'd kind of expect it  to work because you can see that, like a human  
would be able to look at this, and if there are  competent artists, they could turn that into a,   you know, a well-finished product, so you  would expect that a computer could as well.
There are some little tweaks around how it  makes this work which I'll briefly describe  
because we need to be able to go from, kind of,  step 1 through to step 10, through to step 20 and  
so forth. And so, the the way that it does this  it's actually quite clever, what they do is they  
initially, so they take their teacher model, so  remember the teacher model is one that has already  
been trained. Okay so the teacher model already is  a complete Stable Diffusion model that's finished:  
we take that as a given, and we put in our image  —well actually it's noise— we put in our noise,  
right? And we put it through two time steps,  
okay? And then we train our U-Net B —or whatever  you want to call it— to try to go directly from  
the noise to time step number 2. And that's  pretty easy for it to do. And so then what they  
do is they take this —okay, and so this thing  here, remember, is called the student model—.  
They then say, okay, let's  now take that student model  
and treat that as the new teacher. So  they now take their noise and they run  
it through the student model twice, once and  twice, and they get out something at the end.  
And so then, they try to create a new  student —which is a copy of the previous  
student— and it learns to go directly from  the noise to two goes of the student model,  
and you won't be surprised to hear they now take  that new student model and use that to go two goes  
and then they learn, they use that, then they  copy that to become the next student model. And  
so they're doing it again and again and again and  each time they're basically doubling the amount of   work. So it goes one to two, effectively it's then  going two to four, and then four to eight. And  
that's basically what they're what they're doing,  and they're doing it for multiple different time   steps. So the single student model is learning  to both do these initial steps: try to jump  
multiple steps at a time, and it's also letting  to do these later steps: multiple steps at a time.  
And that's it, believe it or not. So this  is this neat paper that came out last week  
and that's how it works. Now, I mentioned that there  was actually two papers.  
The second one is called “On Distillation  of Guided Diffusion Models”. And  
the trick now is, —this second paper, these came  out at basically the same time, if I remember  
correctly, even though they're building each  other from the same teams— is that they say: okay,  
this is all very well but we don't just want to  create random pictures, we want to be able to  
do guidance, right? And you might  remember —I hope you remember from  
last week— that we used something called  Classifier Free Guided Diffusion Models  
which, because I'm lazy, we will just  use an acronym: Classifier Free Guided  
Diffusion Models (CFGDM). And this one, you may  recall, we take, let's say we want a cute puppy,  
we put in the prompt “cute puppy” into our CLIP  text encoder and it spits out an embedding.  
And we put that, —let's ignore the VAE  latents business— we put that into our U-Net,  
but we also put the empty prompt  into our CLIP text encoder;  
we concatenate these things two together   so that then out the other side we get back two  things: we get back the image of the cute puppy  
and we get back the image of some arbitrary  thing —it could be anything—. And then we,  
effectively, do something very much like taking  the weighted average of these two things together,   combine them. And then we use that for  the next stage of our diffusion process.  
Now, what this paper does is it says: this is  all pretty awkward, we end up having to train  
two images instead of one, and, for different  types of levels of guided diffusion, we have to,  
like, do it multiple different times, it's  all pretty annoying, how do we skip it? And,  
based on the description of how we did it before,  you may be able to guess. What we do is we do  
exactly the same student-teacher distillation we  did before, but this time we pass in in addition  
the guidance. And so, again, we've got  the entire Stable Diffusion model, the  
teacher model, available for us, and we are  doing actual CFGD (Classifier Free Guided  
Diffusion) to create our guided diffusion cute  puppy pictures and we're doing it for a range  
of different guidance scales —so you might be  doing 2 and 7.5 and 12 and whatever, right?— and  
those now are becoming inputs to our student  model. So the student model now has additional  
inputs, it's getting the noise, as always,  it's getting the caption or the prompt, —I  
guess I should say— as always, but it's now also  getting the guidance scale. And so, it's learning  
to find out how all of these things are handled  by the teacher model —like what does it do?—  
after a few steps, each time.  So it's exactly the same thing   as before but now it's learning to use the  Classifier Free Guided Diffusion as well.
Okay so, that's got quite a lot going on there  and, if it's a bit confusing, that's okay,  
it is a bit confusing and what I would  recommend is you check out the extra information  
from Johno who has a whole video on this. And one  of the cool things, actually, about this video is  
it's actually a paper walkthrough. And so, part  of this course is, hopefully, we're going to   start reading papers together. Reading papers  is extremely intimidating and overwhelming for  
all of us, all of the time, —at least for me it  never gets any better—. There's a lot of math  
and by watching somebody like Johno, who's an  expert at this stuff, read through a paper,   you'll kind of get a sense of how he  is skipping over lots of the math,  
right?, to focus on, in this case, the really  important thing which is the actual algorithm.   And when you actually look at the algorithm you  start to realize: it's basically all stuff, nearly  
all stuff, maybe all stuff that you did in primary  school or secondary school: so we've got division.  
Okay, sampling from a normal distribution  —so high school— subtraction, division,  
division, multiplication, right? Oh, okay,  we've got a log there, but basically, you know,  
there's not too much going on and then when you  look at the code you'll find it's, you know, once  
you turn this into code, of course, it becomes  even more understandable if you're somebody who's   more familiar with code, like me. So yeah,  definitely check out Johno's video on this.
Explaining “Imagic: Text-Based Real Image Editing with Diffusion Models”
So, another paper came out about three hours ago  and I just have to show you it to you because  
I think it's amazing and so, this is definitely  the first video about this paper because it yeah,  
only came out a few hours ago. But check  this out, this is a paper called “Imagic”.   And with this algorithm you can pass in  an input image —this is just a, you know,  
a photo you've taken or downloaded off the  internet— and then you pass in some text saying:  
“a bird spreading wings”. And what it's going to  try to do is: it's going to try to take this exact   bird, in this exact pose and leave everything as  similar as possible, but adjust it just enough  
so that the prompt is now matched. So here we  take this, this little guy here, and we say:  
oh, this is actually one, we want this to be  a person giving the thumbs up and this is what   it produces. And you can see everything else  is very, very similar to the previous picture.  
So this dog is not sitting but if we  put in the prompt “a sitting dog”,  
it turns it into a sitting dog, leaving  everything else as similar as possible.  
So here's an example of a waterfall and then you  say it's “a children's drawing of a waterfall” and  
now it's become a children's drawing. So, lots  of people in the YouTube chat going: oh my God,   this is amazing, which it absolutely is and  that's why we're going to show you how it works.
One of the really amazing things is you're  going to realize that you understand how it   works already. Just to show you some  more examples, here's the dog image,  
here's the sitting dog, the jumping dog playing  with a toy, jumping dog holding a frisbee.  
Okay, and here's this guy again, giving the  thumbs up, crossed arms, integrating pose to  
namaste hands, holding a cup. So that's pretty  amazing so, I had to show you how this works.  
And I'm not going to go into too much detail but I  think we can get the idea, actually, pretty well.
So what we do is, again, we take, we start  with a fully pre-trained, ready to go  
generative model like a Stable Diffusion model.  And, this is what this is talking about here:  
pre-trained Diffusion model, in the paper they  actually use a model called Imagen but none of   the details, as far as I can see, in any way,  depend on what the model is. It should work just  
fine for Stable Diffusion. And we take “a photo  of a bird spreading wings” okay?, so that's our,  
that's our target and we create an  embedding from that using, for example,  
our CLIP encoder as usual. And we then pass  it through our pre-trained diffusion model.  
And we then see what it creates. And it  doesn't create something that's actually  
like our bird. So then what they do  is they fine-tune this embedding.  
So this is, kind of like, textual inversion,  they fine-tune the embedding to try to make  
the diffusion model output something that's as  similar as possible to the, to the input image.  
And so you can see here they're saying: oh, we're  moving our embedding a little bit. They don't do   this for very long, they just want to move it as  a little bit in the right direction and then now  
they lock that in place, and they say: okay, now  let's fine-tune the entire diffusion model end to  
end, including the VAE —actually with Imagen, and  they have a super resolution model, but same idea—  
so we fine-tune the entire model end to  end, and now the embedding, this optimized   embedding we created… we store in place, we  don't change that at all, that's now frozen.  
And we try to make it so that the diffusion model  now spits out our bird, as close as possible.  
So you fine-tune that for a few epochs and  so you've now got something that takes this  
embedding that we fine-tuned, goes through  a fine-tuned model and spits out our bird.   And then, finally, the original target  embedding we actually wanted —is “a photo  
of a bird spreading its wings”— we ended  up with this slightly different embedding.  
And we take the weighted average of the two,  that's called the interpolate step, the weighted   average of the two. And we pass that through  this fine-tuned diffusion model and we're done.
And so, that's pretty amazing. This would not  take, I don't think, a particularly long time,  
or require any particular special hardware, it's  the kind of thing I expect people will be doing,  
yeah, in the coming days and weeks. But it's very  interesting because, yeah, I mean, the ability to  
take any photo of a person —or whatever—   and change it, like, literally change  what the person's doing is, you know,  
societally very important and really means  that anybody, I guess now, can generate  
believable photos that never actually existed.  I see Jhono in the chat saying that took  
about eight minutes to do it for Imagen  on TPU. Although Imagen's quite a slow,  
big model although the TPUs they  used were very, the latest TPUs. So  
might be a, you know, maybe it's an hour or  something for a Stable Diffusion on GPUs.  
All right, so, that is a lot of fun.
All right, so with that,  let's go back to our notebook.  
Where we left it last time we had, kind  of, looked at some applications that   we can play with in this diffusion-nbs  repo, in the stable_diffusion notebook.  
And what we've got now —and to remind you, when  I say we, it's mainly actually Pedro, Patrick and  
Suraj, just a little bit of help from me—, so  Hugging Face folks. What we-they have done is,  
Stable diffusion pipeline code walkthrough
they're study is they now dig into the pipeline  to pull it all apart step by step, so you can  
see exactly what happens. The first thing I  was just going to mention is: this is how you  
can create those gradual-denoising-pictures, and  this is thanks to something called the Callback.  
So you can say here: when you go through the  pipeline, every 12 steps, call this function;  
and as you can see it's going to call  it with “i” and “t” and the “latents”.   And so, then, we can just make an image  and stick it on the edge end of an  
array, and that's all that's happening  here, right? So this is how you can start  
to interact with a pipeline without  rewriting it yourself from scratch. But now what we're going to do is  we're actually going to write it,  
we're going to, you know, build it from  scratch. So you don't actually have to   use a callback because you'll be able to  change it yourself. So let's take a look.  
So, looking inside the pipeline, what exactly  is going on? So, what's going to be going on   in the pipeline is seeing all of the steps that  we saw in last week's OneNote notes that I drew  
and it's going to be all the code, okay? And we're  not going to show the code of how each step's  
implemented. So, for example, the CLIP text model  we talked about, the thing that takes as input  
a prompt and creates an embedding, we just take  that as a given, okay? So we download it, Open AI  
is trained one called clipped-vit-large-patch-14,  so we just say: from_pretrained, so Hugging Face  
will, Transformers will download and create  that model for us. Ditto for the tokenizer.  
And so ditto for the autoencoder, entered over  the U-Net, so there they all are, we can just  
grab them. So we just take that all as a given,  the… those, these are the three models that  
we've talked about: the text encoder, the clip  encoder, the VAE and the U-Net. So there they are.
So given that we now have those, the next thing  we need is that thing that converts time steps  
into the amount of noise, remember that graph  we drew? And so, we can basically, again,  
use something that Hugging Face —well actually in  this case Catherine Carlson— has already provided,  
which is a scheduler. It's basically  something that shows us that connection,  
so we've got that. So we use that scheduler  and we say how much noise when we're using  
and so we have to make sure that matches, and  so we just use these numbers that we're given.  
Okay, so now, to create our photograph  of astronaut riding a horse, again,  
in 70 steps, with a 7.5 guidance scale, batch size  of 1. Step number one is to take our prompt and  
tokenize it. Okay, so, we looked at that in Part  1 of the course, so check that out if you can't  
remember what tokenizing does, but it's just  splitting it in, basically, it's splitting it   into words or sub-word units if they're long and  unusual words. So here are… so this will be the  
start, the start of sentence token, and this  will be our “ a photograph of an astronaut”,  
etc. And then you can see the same token  is repeated again, again at the end, that's   just the padding to say: we're all done. And the  reason for that is that GPUs and TPUs really like  
to do lots of things at once, so we kind of have  everything be the same length by padding them.  
That may sound like a lot of wasted work, which  it kind of is, but a GPU would rather do lots of  
things at the same time, on exactly the same sized  input, so this is why we have all this padding.
So you can see here, if we decode that  number, it's the end of text marker,  
which is just padding, really, in this case. As  well as getting the input IDs —so these are just  
lookups into a vocabulary— there's also a mask,  which is just telling it which ones are actual  
words as opposed to padding —which is not very  interesting—. So we can now take those input IDs,  
we can put them on the GPU, and we  can run them through the CLIP encoder.  
And so, for a batch size of one, so we've got one  image, that gives us back a 77x768, because we've  
got 77 here and each one of those creates a 768  long vector, so we've got a 77 by 768 tensor. So  
these are the embeddings for “a photograph of an  astronaut riding a horse” that come from CLIP. So  
remember, everything's pre-trained so that's  all done for us, we're just doing inference.  
And so remember for the classifier free guidance,  we also need the embeddings for the empty string,  
so we do exactly the same thing.  
So now we just concatenate those  two together, because we're just,   this is just a trick to get the  GPU to do both at the same time,  
because we like the GPU to do as  many things at once as possible.
And so now we create our noise.   And because we're doing it with a VAE, we can  call it latents, but it's just noise, really.  
I wonder if you'd still call it that without the  VAE, maybe you would have to think about that.  
So that's just random numbers, normally generated,  normally distributed random numbers of size one  
—that's our batch size— and the reason that we've  got this divided by 8 here is because that's what  
the VAE does, it allows us to create things that  are eight times smaller by height and width,  
and then it's going to expand it up again for  us later, that's why this is so much faster.  
You'll see a lot of this, after we put it  on the GPU you'll see a lot of this .half(),   this is converting things into what's called half  precision or fp_16. Details don't matter too much,  
it's just making it half as big in memory by  using less precision. Modern GPUs are much,  
much, much faster if we do that, so you'll see  that a lot. If you use something like fastai,  
you don't have to worry about it, all this  stuff is done for you. Then we'll see that   later as we rebuild this with much,  much less code later in the course.  
So we'll be building our own  kind of framework from scratch,  
which you'll then be able to  maintain and work with yourself. Okay, so we have to say we want to do 70 steps.  
Something that's very important —we won't  worry too much about the details right now—,   but this, what you'll see here is that we take our  random noise and we scale it. And that's because  
Scaling random noise to ensure variance
depending on what stage you're up to, you need  to make sure that, kind of, you have the right  
amount of variance, basically, otherwise you're  going to get activations and gradients that go  
out of control. This is something we're going  to be talking about a huge amount during this   course and we'll show you lots of tricks to  handle that kind of thing, automatically.
Unfortunately, at the moment, in the Stable  Diffusion world this is all done in rather,  
in my opinion, kind of, ways that are too tied to  the details of the model. I think we'll be able  
to improve it as the course goes on, but for now  we'll stick with how everybody else is doing it,  
this is how they do it. So we're going to be  jumping through… so normally it would take  
a thousand time steps but because we're  using a fancy scheduler we get to skip,   from 999 to 984. 984 to 970 and so forth. So we're  going down about 14 time steps. And remember this  
is a very, very unfortunate word. They're not time  steps at all, in fact they're not even integers.  
It's just a measure of how much noise are we  adding at each time, and you find out how much  
noise by looking it up on this graph, okay?  That's all time step means, it's not a step  
of time and it's a real shame that that word  is used because it's incredibly confusing.  
This is much more helpful, this is the actual  amount of noise at each one of those iterations.  
And so here you can see the amount of noise for  each of those time steps and we'll be going,  
we're going to, be going backwards as  you can see we start at 999. So we'll   start with lots of noise and then we'll be  using less and less and less and less noise.
So we go through the 70 time steps, in a for loop,  
concatenating our two noise bits  together because we've got the   classifier free and the prompt versions. Do our  scaling, calculate our predictions from the U-Net.  
And notice here we're passing in  the time step as well as our prompt.  
That's going to return two things: the  unconditional prediction, so that's the one for  
the empty string —remember we passed in, one of  the two things we passed in was the empty string—  
so we concatenated them together and so  after they come out of the U-Net, we can  
pull them apart again, so .chunk() just means:  pull them apart into two separate variables.  
And then we can do the guidance  scale that we talked about last week.  
And so now we can do that update,  where we take a little bit of the noise  
and remove it to give us our  new latents. So that's the loop.
And so at the end of all  that, we decode it in the VAE  
—the paper that created this VAE tells us that  we have to divide it by this number to scale  
it correctly— and once we've done that, that gives  us a number which is between negative one and one.  
Python Imaging Library expects  something between zero and one,   so that's what we do here, to make it between  zero and one, and like enforce that to be true,  
put that back on the CPU, make sure it's, that  the order of the dimensions is the same as what  
Python Imaging Library expects. And then finally  convert it up to between 0 and 255 as an INT,  
which is actually what PIL really wants and  there's our picture. So there's all the steps.
So. What I then did, this is kind of  like… so the way I normally build code,  
I use notebooks for everything, is I kind of do  things step by step by step, and then I tend to,  
kind of, copy them. And I use shift M —I  don't know if you've seen that— but what   shift M does it takes two cells and combines  them, like that. And so, I basically combined  
some of the cells together and I removed a bunch  of the… the pros, so you can see the entire thing  
on one screen. And what I was trying to do here  is: I'd like to get to the point where I've got  
something which I can very quickly do experiments  with. So maybe I want to try some different   approach to guidance free classification,  maybe I want to add some callbacks.  
So on and so forth. So I kind of like to have  everything, you know, I like to have all of the,  
my important code, be able to fit into my  screen at once. And so you can see now I do,  
I've got the whole thing on my screen so  I can keep it all in my head. One thing I   was playing around with was I was trying  to understand the actual guidance free  
equation in terms of, like, how does it work?  Computer scientists tend to write things —and  
software engineers— with kind of long words as  variable names; mathematicians tend to use short  
words, just letters, normally. For me, when  I want to play around with stuff like that,   I turn stuff back into letters and that's because  I actually kind of pulled out one note and I  
started jutting down this equation and playing  around with it to understand how it behaves.  
So this is just, like, it's not better or worse,  it's just depending on what you're doing. So   actually here I said: okay, “g” is called this  guidance scale. And then, rather than having  
the unconditional and text embeddings, I just  call them “u” and “t”. And now I've got this all   down into an equation which I can write down in  a notebook and play with and understand exactly  
how it works. So that's something I find really  helpful for working with this kind of code, is to,  
yeah, turn it into a form that I can manipulate  algebraically more easily. I also try to make it  
look as much like the paper that I'm implementing  as possible. Anyways that's that code.
So then I copied all this again and I basically   —oh, I actually did it for two prompts, this  time—. I thought this was fun: “an oil painting  
of an astronaut riding a horse in the style  of Grant Wood”. Just to remind you, Grant Wood  
looks like this. Not obviously astronaut material  which I thought would make it actually, kind of,  
particularly interesting. Although it does have  horses. I can't see one here, some of his pictures  
have horses. So because I did two prompts, I  got back two pictures I could do. So here's  
the Grant Wood one. I don't know what's going on  in his back here, but I think it's quite nice.  
So yeah, I then copied that whole thing again  and merged them all together and then just put it  
into a function, so I took the little bit which  creates an image and put that into a function,  
I took the bit which does the tokenizing and  text encoding and put that into a function. And so, now, all of the code necessary to do the  whole thing from, you know, top to bottom fits  
in these two cells. Which makes it, for me,  much easier to see exactly what's going on.  
So you can see I've got the text embeddings,  I've got the unconditional embeddings,  
I've got the embeddings which concatenate  the two together, optional random seed,  
my latents. And then the loop itself. And  you'll also see a, something I do which is  
a bit different to a lot of software engineering  is I often create things which are, kind of like,   longer lines, because I try to  have each line be, kind of, like,  
mathematically one thing that I want  to be able to think about as a whole.   So yeah, these are some differences between,  kind of, the way I find numerical programming  
works well compared to the way I would write a  more traditional software engineering approach.  
And again, this is partly a personal preference  but it's something I find works well for me. So we're now at a point where we've got,  yeah, two fun… three functions that easily  
fit on the screen and do everything.  So I can now just say: make samples   and display each image. And so  this is something for you to  
experiment with and what I specifically suggest  as homework is to try picking one of the  
Recommended homework for the week
extra tricks we learned about like image to image,  or negative prompts. Negative prompts would be a  
nice easy one. Like, see if you can implement  negative prompt in your version of this. Or,  
yeah, try doing image to image. That wouldn't be  too hard either. Another one you could add is try  
adding callbacks. And the nice thing is then, you  know, you've got code which you fully understand  
because you know what all the lines do. And you  then don't need to wait for the diffusers folks  
to update it —the library— to do is, for example,  the callbacks are only added like a week ago so,  
until then, you couldn't do callbacks.  Well now you don't have to wait for the   diffusers team to add something. The  code's all here for you to play with.  
So that's my recommendation as  a bit of homework for this week.
Okay, so, that brings us to the end of our rapid  overview of Stable Diffusion and some very recent  
papers that very significantly developed Stable  Diffusion. I hope that's given you a good sense of  
the kind of very high level, slightly  hand wavy version of all this and you   can actually get started playing with some  fun code. What we're going to be doing next  
is going right back to the start, learning how  to multiply two matrices together, effectively,  
and then gradually building from there until we've  got to the point that we've rebuilt all this from   scratch and we understand why things work the  way they do, understand how to debug problems,  
improve performance and implement new research  papers, as well. So that's going to be very  
exciting and so, we're going to have a break  and I will see you back here in 10 minutes.
Okay, welcome back everybody. I'm really  excited about the next part of this.  
It's going to require some serious  tenacity and a certain amount of patience,  
but I think you're going to learn a lot. A  lot of folks I've spoken to have said that  
previous iterations of this part of the course  is like the best course they've ever done and  
this one's going to be dramatically better  than any previous version we've done of this.   So, hopefully, you'll find the  hard work and patience pays off.
We're working now through the course22p2  repo —so 2022 course, Part 2—.  
And the notebooks are ordered. So  I start with notebook number one.  
And, okay, so the goal is to get to  Stable Diffusion from the foundations,  
What are the foundations of stable diffusion? Notebook deep dive
which means we have to define what  are the foundations. So I've decided  
to define them as follows: we're allowed to  use Python. We're allowed to use the Python  
standard library, so that's all the stuff that  comes with Python by default. We're allowed to  
use Matplotlib because I couldn't be bothered  creating my own plotting library, and we're  
allowed to use Jupyter notebooks and nbdev which  is something that creates modules from notebooks.
So basically what we're going to try to do  is to, yeah, rebuild everything starting  
from this foundation. Now, to be clear, what…  we are allowed to use are the libraries once  
we have re-implemented them correctly and so,  if we, if we re-implement something from NumPy  
or from Pytorch or whatever, we're then allowed  to use the NumPy or Pytorch or whatever version.  
Sometimes we'll be creating things that  haven't been created before and that's   then going to be becoming our own library. And  we're going to be calling that library “mini-ai”.  
So we're going to be building our  own little framework as we go. So, for example, here are some imports and these  Imports all come from the Python standard library  
except for these two. Now, to be  clear, one challenge we have is that  
the models were used in Stable Diffusion were  trained on millions of dollars worth of equipment  
for months, which we don't have the time or money  so, another trick we're going to do is we're going  
to create smaller —identical but smaller— versions  of them. And so once we've got them working,  
we'll then be allowed to use the big pre-trained  versions. So that's the basic idea. So we're  
going to have to end up with our own VAE, our  own U-Net, our own CLIP encoder and so forth.
To some degree I am assuming that  you've completed Part 1 of the course,   to some degree. I will cover everything,  at least briefly, but if I cover something  
about Deep Learning too fast for you to  know what's going on and you get lost,  
you know, go back and and watch Part 1,  or go and, you know, Google for that term.  
For stuff that we haven't covered in Part 1 I  will go over it very thoroughly and carefully.  
All right, so, I'm going to assume that you   know the basic idea that which is that we're going  to need to be doing some matrix multiplication.  
So we're going to try to take a deep dive into  matrix multiplication today and we're going to   need some input data and I quite like working  with MNIST data. MNIST is handwritten digits.  
It's a classic data set. They're  28 by 28 pixel, grayscale images.
And so we can download them from this URL. So we  use the pathlib Path object a lot, it's part of  
Python and it basically takes a string and turns  it into something that you can treat as a path,   for example, you can use slash to mean: this  file inside this subdirectory. So this is  
how we create a Path object. Path objects  have, for example, a make directory method.  
So I like to get everything set up, but I want  to be able to rerun this cell lots of times and  
not have it, like, give me errors if I run  it more than once, so if I run it a second   time it still works, in that case that's  because I put this exist okay equals true.  
How did I know that I can say… because, otherwise,  it would try to make the directory, it would   already exist and it'll give an error. How do I  know what parameters I can pass to make there:  
I just press shift tab. And so when I hit  shift tab it tells me what options there are.  
If I press it a few times it'll actually  pop it down to the bottom of the screen   just to remind me. I can press escape  to get rid of it. Or, you can just…  
or else you can just hit tab inside and it'll list  all the things you could type here as you can see.
All right, so, we need to grab this URL and so  Python comes with something for doing that which  
is the urllib library —that's part of Python— that  has something called urlretrieve. And something  
which I'm always a bit surprised is not widely  used is people reading the Python documentation,  
so you should do that, a lot. So if I  click on that, here is the documentation  
for urlretrieve. And so I can find exactly what  it can take and I can learn about exactly what it  
does and so I, yeah, I read the documentation from  the Python docs for every single method I use and  
I look at every single option that it takes and  then I practice with it. And to practice with it   I practice inside Jupiter. So if I want this  input on its own I can hit Ctrl Shift Hyphen  
and it's going to spit it into two cells, and then  I'll hit Alt Enter or Option Enter so I can create  
something underneath, and I can type urlretrieve,  Shift Tab and so there is, there it all is.  
If I'm like way down somewhere in the notebook  and I have no idea where urlretrieve comes from,  
I can just hit Shift Enter and it actually  tells me exactly where it comes from.  
And if I want to know more about it, I can  just hit question mark (“?”) shift enter   and it's going to give me the documentation. And  most call of all, second question mark (“??”)  
and it gives me the full source code. And you  can see it's not a lot, you know, reading the  
source code of Python standard library stuff is  often quite revealing and you can see exactly  
how they do it, that's a great way  to learn more about… more about this.
So in this case I'm just going to use a very  simple functionality, which is I'm going to   say the URL to retrieve and the file name to  save it as. And again, I made it so I can run  
this multiple times so it's only going to do the  urlretrieve if the path doesn't exist. If I've  
already downloaded it I don't want to download  it again. So I run that cell and notice that I   can put exclamation mark (“!”) followed by line  of Bash and it actually runs this using Bash.  
If you're using Windows this won't work  and I would very, very strongly suggest  
if you're using Windows use WSL and if you use  WSL all of these notebooks will work perfectly.  
So, yeah, do that. All routed on Paper Space or  Lambda Labs or something like that, Colab, etc.
Okay, so this is a gzip file so, thankfully,  Python comes with a gzip module —Python comes  
with quite a lot actually— and so we can open  a gzip file using gzip.open() and we can pass  
in the path, then we say we're going to  read it as binary —as opposed to text—.  
Okay, so this is called a Context Manager it's  a, it's a “with” clause and what it's going to  
do is it's going to open up this gzip file, the  gzip object will be called “f” and then it runs  
everything inside the block and when it's done it  will close the file. So “with” blocks can do all  
kinds of different things but in general “with”  blocks that involve files are going to close the  
file automatically for you. So we can now do that  and so, you can see, it's opened up the gzip file  
and the gzip file contains what's called  pickle objects. Pickled objects is, basically,  
Python objects that have been saved to disk. It's  the main way that people in pure Python save stuff  
and it's part of the standard library. So this  is how we load in from that file. Now the file  
contains a couple of tuples, so when you put a  tuple on the left hand side of an equal sign it's  
quite neat, it allows us to put the first tuple  into two variables called x_train and y_train  
and the second into x_valid and y_valid. This  trick here where you put stuff like this on  
the left is called “destructuring” and it's  a super handy way to make your code kind of  
clear and concise and lots of languages  support that including Python.
Okay. So we've now got some data and so we can  have a look at it. Now, it's a bit tricky because  
we're not allowed to use NumPy —according to our  rules— but unfortunately this actually comes as   NumPy, so I've turned it into a list. All right,  so I've taken the first image and I've turned it  
into a list and so we can look at a few examples  of some values in that list. And here they are.  
So it looks like they're numbers between zero  and one. And this is what I do, you know, when I  
learn about a new data set. So when I started  writing this notebook. What you see here  
—other than the prose here— is what I actually  did when I was working with this data:  
is I wanted to know what it was, so I just  grab a little bit of it and look at it.  
So I kind of got a sense now of what it is. Now, interestingly,  
it's 784 —this image— is 784 long list.  
Oh dear, people freaking out in the comments: no  NumPy! Yeah, no NumPy. Do you see NumPy? No NumPy.
Why 784? What is that? Well that's  because these are 28 by 28 images.  
So it's just a flat list here of 784 long.  So, how do I turn this 784 long thing into  
28 by 28? So I want a 28, list of 28 lists of  28, basically, because we don't have matrices.  
So how do we do that? And so we're going to be  learning a lot of cool stuff in Python here.  
I'm sorry I can't stop laughing at all this stuff  in our chat… oh dear, people are quite reasonably  
freaking out, that's okay, we'll get there, I  promise. I hope —otherwise I'll embarrass myself.
Numpy arrays and PyTorch Tensors from scratch
All right, so how do I convert a 784 long  list into 28 lists… 28 long list of 28 long  
lists. I'm going to use something called “chunks”  and first of all I'll show you what this thing  
does and then I'll show you how it works.  So “vals” is currently a list of 10 things.  
Now if I take “vals” and I pass it to  chunks() with 5 it creates two lists of  
five. Here's list number one of five elements,  and here's list number two of five elements.  
Hopefully you can see what it's doing: it's  chunkifying this list and this is the length   of each chunk. Now, how did it do that? The way  I did it is using a very, very useful thing in  
Python that far too many people don't know about  which is called “yield”. And what “yield” does is,  
you can see here I've got a loop, it's going  to go through from zero up to the length of my  
list and it's going to jump by five at a time.  That's going to go, in this case, 0 comma 5.  
And then it's going to —think of this as being  like return, for now— it's going to return   the list from 0 up to 5. So it returns the first  bit of the list. But yield doesn't just return,  
it kind of like returns a bit and then  it continues, and it returns a bit more.  
And so specifically what yield does is it creates  an iterator. An iterator is, an iterator is  
basically something you can, well, this actually  let's use it, that you can call “next” on a bunch  
of times. So let's try it, so we can say iterator  equals (val_iter =)... Okay, oh, I'm gonna run it.
So, what is iterator? Well iterator (val_iter)  is something that I can basically, I can call   “next” on. And next basically says: yield the  next thing. So this should yield vals[0:5],  
there it is, it did right. There's vals[0:5].  Now, if I run that again it's going to give me  
a different answer because it's now  up to the second part of this loop,  
now it returns this, the last five. Okay, so,  
this is what our iterator does. Now, if  you pass an iterator to Python's list  
it runs through the entire  (iter_val) iterator until   it's finished and creates a list of the  results. And what is finished looks like?  
This is what finish looks like, if you call next  and get stop iteration, that means you've run out,  
and that makes sense, right?, because  my loop, there's nothing left in it.  
So, all of that is to say, we now have a  way of taking a list and chunkifying it.  
So what if I now take my full image, image number  one, chunkify it into chunks of 28 long and turn  
that into a list and plot it: ta-da! We have  successfully created an image. So that's good.
Now, we are done, but there are other ways to  create this iterator. And because iterators  
and generators —which are closely related— are so  important, I wanted to show you more about how to  
do them in Python. It's one of these things that,  if you understand this, you'll often find that you  
can throw away huge pieces of enterprise software  and basically replace it with an iterator,  
that lets you stream things one bit at a  time, it doesn't store it all in memory,  
it's this really powerful thing that once, I often  find, once I show it to people they suddenly go 
like: oh! wow! you know, we've been using all this   third-party software and we could  have just created a Python iterator.  
Python comes with a a whole standard library  module called “itertools” just to make it easier  
to work with iterators. I'll show you one example  of something from itertools which is “islice”.
So let's grab our values, again, these 10 values.  
Okay. So let's take these 10 values and we can  take any list and turn it into an iterator by  
passing it to iter() —which I should call “it”—.  That so I don't override this Python… that's not a  
keyword, but this thing I don't want to override.  So, this is now basically something that I can  
call —actually let's do this, I'll show you that  I can call— next on it. So if I now go next(it)...  
you can see it's giving me  each item, one at a time.   Okay so that's what converting it into an  iterator does. “islice” converts “it” into  
a different kind of iterator, let's  call this maybe islice iterator (isit).  
And  
so you can see here, what it did, was it jumped…  
stop, here we are, so, ah yes that's  what have been better. So I should   query create the iterator and then call next a  few times —sorry, this is what I meant to do—.  
It's now only returning the first five,  before it calls stop iteration —before  
it raises stop iteration—. So what  islice does is it grabs the first  
N things from an iterable —something that  you can iterate—. Why is that interesting?  
Because I can pass it to list(), for example,  
right? and now, if I pass it to list again this  iterator has now grabbed the first five things,  
so it's now up to thing number  six, so if I call it again,   it's the next five things, and if I call  it again, then there's nothing left.  
And maybe you can see we've actually now got  this defined but we can do it with islice,  
and here's how we can do it.  It's actually pretty tricky.
“iter”, in Python, you can pass it something  like a list —to create an iterator— or you can  
pass it —now this is a really important word—  a “callable”. What's a callable? A callable is,  
generally speaking, it's a function, it's  something that you can put parentheses after.  
Could even be a class, anything you  can put parentheses after. You can   just think of it for now as a function.  So we're going to pass it a function  
and, in the second form, it's going to be called  until the function returns this value here,  
which in this case is empty list. And we just saw  that islice will return empty list when it's done.  
So this here is going to  keep calling this function  
again and again and again, and we've seen exactly  what happens because we've called ourselves  
before, there it is, until it gets an empty list.  So if we do it with 28 then we're going to get  
our image again. So we've now got two different  ways of creating exactly the same thing.  
And if you've never used iterators before,  
now's a good time to pause the video and  play with them, right? So, for example,  
you could take this here, okay? and if you've  not seen lambdas before they're exactly the  
same as functions, but you can define them in  line, so let's replace that with a function.
Okay, so now I've turned it into a function  and then you can experiment with it.  
So let's create our iterator, then call f() on it…  
and you can see there's the first 28. And  each time I do it I'm getting another 28.  
Now the first two rows are all empty but finally,  look, now I've got some values. Call it again.   See how each time I'm getting something  else, just calling it again and again.  
And that is the values in our iterator. So  that gives you a sense of like how you can  
use Jupiter to experiment so, what you should do  is, as soon as you hit something in my code that  
doesn't look familiar to you, I recommend  pausing the video and experimenting with that in  
Jupiter. And, for example, iter(), most people  probably have not used it at all and certainly  
very few people have used this two argument  form, so hit shift tab a few times and now   you've got at the bottom there's a description  of what it is. Or find out more: Python iter.  
Here we are, go to the docs. Well, that's not  the right bit of the docs. See API, wow! crazy,  
that's terrible. Let's try  searching here: iter. Here  
we go. That's more like it. So now you've got  links so if it's, like, okay, it returns an  
iterator object, what's that? Well click on it,  find out, now, this is really important to know   and here's that stop exception that we saw: so,  stop iteration exception. We saw next already,  
we can find out what iterable is. And here's  an example. And as you can see it's using  
exactly the same approach that we did, but  here it's being used to read from a file:   this is really cool! Here's how to read from a  file, 64 bytes at a time, until you get nothing,  
processing it, right? So the the  docs of Python are quite fantastic,  
as long as you use them. If you don't  use them they're not very useful at all.  
And I see Sifa in the comments,  our local Haskell programmer,  
appreciating this Haskellness in Python,  so that's good. It's not quite Haskell,  
I'm afraid, but it's the  closest we're going to come. All right, how are we going for time? Pretty good.
Okay, so now that we've got image, which is  a list of lists, and each list is 25(?) long,  
we can index into it, so we can say  image 20 —well let's do it—. Image 20.  
Okay, is a list of 28 numbers and  then we could index into that.  
Okay, so we can index into it. Now, normally  we don't like to do that for matrices,  
we would normally rather write it like this,   okay? So that means we're going to have  to create our own class to make that work.  
So, to create a class in Python, you write “class”  and then you write the name of it. And then you  
write some really weird things. The weird things  you write have two underscores, a special word,  
and then two underscores. These things with  two underscores on each side are called Dunder  
methods and they're all the special, magically  named, methods which have particular meanings  
to Python. And you're just going to let them, but  they're all documented in the Python object model.  
__init__ object model.  
Okay finally. Okay so once you eventually find,  oh, it's called data model not object model.  
And so this is basically where all  the documentation is about absolutely   everything and I can click dunder init  (__init__) and it tells you, basically,  
this is the thing that constructs objects.  So anytime you want to create a class,  
that you want to, that you want to construct  and it's going to store some stuff,   so in this case it's going to store our image,  you have to define dunder init (__init__).
Python is slightly weird in that every  method, you have to put “self” here —for  
reasons we probably don't really need  to get into right now— and then any   parameters. So we're going to be creating an  image passing in the thing to store the “xs”,  
we're going to be passing in the “xs”. And so  here we're just going to store it inside the self.  
So once I've got this line of code, I've now  got something that knows how to store stuff,  
the xs inside itself. So now I want to be  able to call square bracket 20 comma 15  
([20,15]). So how do we do that? Well, basically,  part of the data model —there's a, there's a  
special thing called dunder getitem (__getitem__).  And when you call square brackets on your object  
that's what Python uses. And it's going to  pass across the [20,15] here, that's indices.  
So we're now, basically, just going  to return this, so the self.xs,  
with the first index and the second index.   So let's create that Matrix class and run  that and you can now see m[20,15] is the same.  
Oh, quick note on, you know, ways in which  my code is different to everybody else's   —which it is,— it's somewhat unusual to put  definitions of methods on the same line, as  
the the signature, like this, I do it quite a lot  for one-liners. As I've kind of mentioned before,  
I find it really helps me to be able to see all  the code I'm working with, on the screen at once.  
A lot of the world's best programmers actually  have had that approach as well. It seems to work   quite well for some people that are extremely  productive. It's not common in Python —some  
people are quite against it— so, if you're at work  and your colleagues don't write Python this way,  
you probably shouldn't either. But if you can  get away with it I think it works quite well. 
Anywho, okay, so now that we've created something  that lets us index into things like this, we're   allowed to use Pytorch, because we're allowed  to use this one feature in Pytorch. Okay, so,  
we can now do that and so now, to create a  tensor, which is basically a lot like our Matrix,  
we can now pass a list into tensor to  get back a tensor version of that list,  
or perhaps, more interestingly, we could pass in  a list of lists —maybe let's give this a name—  
Oopsie Daisy: that needs to be a list of lists,  
just like we had before for our image. In fact,  let's do it for our image: let's just pass in  
our image, there we go. And so now  we should be able to say tens[20,15],  
and there we go. Okay, so, we've  successfully reinvented that.
All right. So now we can convert all of our lists  into tensors. There's a convenient way to do this,  
which is to use the map function  in the Python standard library.  
So shift… shift tab, map takes a function and  then some iterables, in this case one iterable,  
and it's going to apply this function to each of  these four things and return those four things,  
and so then I can put four things on the left  to receive those four things. So this is going  
to call tensor x_train and put it in x_train.  Tensor y_train put into y_train and so forth.
So this is converting all of these lists to  tensors and storing them back in the same name.  
So you can see that x_train now is a tensor,  so that means it has a shape property: it has  
50,000 images in it, which are each 784  long. And you can find out what kind of…  
what kind of stuff it contains by calling it  .type(), so it contains floats. So this is the  
tensor class, we'll be using a lot of it so,  of course, you should read its documentation.  
I don't love the Pytorch documentation, some of  it's good, some of it's not good, it's a bit all  
over the place. So here's tensor, but it's well  worth scrolling through to get a sense of, like,  
this is actually not bad, right? it tells you how  you can construct it, this is how I constructed   one before, passing it list of lists. You can  also pass it NumPy arrays. You can change types,  
so on and so forth. So, you know, it's  well worth reading through and like,   you're not going to look at every single  method it takes, but you're kind of,  
if you browse through it, you'll get  a general sense, right? That tensors  
do just about everything you couldn't  think of for a numeric programming.  
At some point you will want to know every single  one of these, or at least be aware roughly what  
exists so you know what to search for in the docs,  otherwise you will end up recreating stuff from  
scratch, which is much much slower than simply  reading the documentation to find out it's there.
All right, so, instead of calling chunks, or  islice, the thing that is roughly equivalent  
in a tensor is the reshape method. So reshape.  So to reshape our 50,000 by 784 thing, we can,  
we want to turn it into 50,000 28 by 28 tensors.  So I could write here reshape(50,000, 28, 28),  
but I kind of don't need to because I  could just put -1 here and it can figure  
out that that must be 50… that that must be  50,000, because it knows that I have 50,000   by 784 items. So it can figure out, so -1  means just fill this with all the rest.
Okay, now, what does the word  
tensor mean?  
So there's some very interesting history here and  I'll try not to get too far into it because I'm a  
bit over enthusiastic about this stuff, I must  admit. I'm very, very interested in the history   of tensor programming, and array programming, and  it basically goes back to a language called APL.  
History of tensor programming
APL is a… basically originally a mathematical  notation that was developed in the mid to late  
50s, 1950s and, at first, it was used to, as a  notation for defining how certain new IBM systems  
would work. So it's all written out in this, in  this notation, It's kind of like a replacement  
for mathematical notation that was designed to  be more consistent and kind of more expressive.  
In the early 60s —so the guy who wrote made  it was called Ken, his name Ken Iverson—,  
in the early 60s some implementations  that actually allowed this notation to  
be executed on a computer appeared, both the  notation and the executable implementations,  
slightly confusingly, are both called  APL. APL's been in constant development,  
ever since that time, and today is one of the  world's most powerful programming languages  
and you can try it by going to try APL.  And why am I mentioning it here, because  
one of the things Ken Iverson did, well, he  studied an area of physics called tensor analysis.  
And, as he developed APL, he basically said,  like, oh, what if we took these ideas from tensor  
analysis and put them into a programming  language. So in, yeah, in APL you could,  
you can and, you know, have been able to for some  time, can basically, you can define a variable  
and rather than saying equals —which is a terrible  way to define things, really mathematically,  
because that has a very different meaning most  of the time in math— instead we use arrow to  
define things. We can say, okay, that's going  to be a tensor. Like so. And then we can look  
at the contents of “a” and we can do things  like, oh, what if we do a x 3, or a - 2,  
and so forth. And as you can see, what it's doing  is it's taking all the contents of this tensor  
and it's multiplying them all by 3, or subtracting  2 from all of them. Or perhaps, more fun,  
we could put into “b” a different tensor and  we can now do things like “a” divided by “b”  
and you can see it's taking each  of “a” and dividing by each of “b”.
Now, this is very interesting because,  now, we don't have to write loops anymore.  
We can just express things directly. We can  multiply things by scalars even if they're,  
this is called a rank one tensor, that is to  say, it's basically, in math we call it a vector.  
We can take two vectors and can divide one by the  other, and so forth, it's a really powerful idea.  
Funnily enough APL didn't call them  tensors even though Ken Iverson  
said he got this idea from tensor  analysis. APL calls them arrays.  
NumPy, which was heavily influenced  by APL, also caused them arrays.  
For some reason Pytorch, which is very heavily  influenced by APL —sorry, by NumPy— doesn't call  
them arrays, it calls them tensors. They're all  the same thing. They are rectangular blocks of  
numbers. They can be one-dimensional, like  a vector. They can be two-dimensional like  
a matrix. They can be three-dimensional,  which is like, a bunch of stacked matrices,   like a batch of matrices and so forth. If you  are interested in APL, which I hope you are,  
we have a whole APL and array programming  section on our forums and also we've prepared  
a whole set of notes on every single glyph in  APL, which also covers all kinds of interesting  
mathematical concepts, like, complex direction  and magnitude and all kinds of fun stuff like  
that. That's all totally optional but a lot of  people who do APL say that they feel like they've  
become a much better programmer in the process  and also you'll find here at the forums a set of  
17 study sessions of an hour or two each covering  the entirety of the language, every single glyph.  
So that's all, like, where this stuff comes from.
So this batch of 50,000 images. 50,000 28 by  28 images is what we call a rank three tensor  
in Pytorch. In NumPy we would call it an array  with three dimensions, those are the same thing.  
So, what is the rank? The rank is  just a number of dimensions. It's   50,000 images of 28 high by 28 wide, so there  are three dimensions, that is the rank of the  
tensor. So, if we then pick out a particular  image, right? Then we look at its shape.  
We could call this a matrix, it's a 28 by 28  tensor or we could call it a rank 2 tensor.  
A vector is a rank 1 tensor. In APL a  scalar is a rank 0 tensor and that's  
the way it should be. A lot of  languages and libraries don't,   unfortunately, think of it that way. So what  is a scalar is a bit dependent on the language.
Okay, so we can index into the 0th image, 20th  row, 15th column to get back this same number.  
Okay, so we can take x_train.shape,  
which is 50,000 by 784, and you  can destructure it into “n”,  
which is the number of images, and “c”, which  is the number of, the full number of columns,  
for example. And we can also… well this  is actually part of the standard library,  
so we're allowed to use min() so we can find  out in y_train what's the smallest number,  
and what's the maximum number. So they go from 0  to 9. So you see here, it's not just the number 0,  
it's a scalar tensor 0. They act almost the same,  most of the time. So here's some example of a bit  
of the y training, the bit of y_train, so you can  see these are basically— this is going to be the  
labels, right? These are our digits. And this is  its shape so there's just 50,000 of these labels.
Okay and so since we're allowed to use this in the  standard library well, it also exists in Pytorch,  
so that means we're also allowed to use the  .min() and .max() properties. All right, so,  
Random numbers from scratch
before we wrap up we're going to do one more thing  and I don't know what the —we would call, kind of,  
anti-cheating— but according to our rules we're  allowed to use random numbers, because there is  
a random number generator in the python standard  library but, we're going to do random numbers from  
scratch ourselves. And the reason we're going to  do that is, even though according to the rules we  
could be allowed to use the standard library, one:  it's actually extremely instructive to build our  
own random number generator from scratch. Well,  at least I think so, let's see what you think.
So, there is no way, normally, in  software, to create a random number…  
unfortunately. Computers, you know, add,  subtract, times, logic gates —stuff like that—,  
So, how does one create random numbers? Well,   you could go to the Australian National  University quantum random number generator  
and, this, looks at the quant fluctuations of the  vacuum and provides an API which will actually  
hook you in and return quantum random  fluctuations of the vacuum. So that's about,  
that's the most random thing I'm aware of, so  that would be one way to get random numbers.  
And there's actually an API for that —so there's  a bit of fun—. You could do what Cloudflare does.  
Cloudflare has a huge wall full of lava lamps and  it uses the pixels of a camera looking at those  
lava lamps to Generate random numbers. Intel,  nowadays, actually has something in its chips  
which you can call, RDRAND, which will return  random numbers on certain Intel chips from 2012.
All of these things are, kind of, slow.  They can, kind of, get you one random   number from time to time. We want some way  of getting lots and lots of random numbers.  
And so what we do is we use something  called a pseudo-random number generator.  
Pseudo-random number generator is a mathematical  function that you can call lots of times  
and each time you call it, it will  give you a number that looks random.  
To show you what I mean by that,  I'm going to run some code.  
And I've created a function —which  we'll look at in a moment— called “rand”   and if I call rand() 50 times and plot it,  there's no obvious relationship between one  
call and the next. That's one thing that I  would expect to see from my random numbers,  
I would expect that each time I call rand(), the  numbers would look quite different to each other.  
The second thing is: rand() is meant  to be returning uniformly distributed  
random numbers and therefore, if I call  it lots and lots and lots of times,   and plot its histogram, I would expect to see  exactly this, which is each, from 0 to 0.1  
there's a few, from 0.1 to 0.2 there's a few.  From 0.2 to 0.3 there's a few. It's a fairly  
evenly spread thing. These are the two key things  I would expect to see. An even distribution of  
random numbers and that there's no correlation,  or no obvious correlation, from one to the other.
So we're going to try and create a function  that has these properties. We're not going to  
derive it from scratch. I'm just going to tell  you that we have a function here called the   Wickman-Hill algorithm. This is actually what  Python used to use back in before python 2.3.  
And the key reason we need to know about  this is to understand, really well,   the idea of random state. Random state is  a global variable, it's something which is,  
or at least it can be —most of the time when we  use it we use it as a random variable— and it's  
just basically one or more numbers. So we're  going to start with no random state at all,  
we're going to create a function called  “seed” that we're going to pass something   to. And I just mashed the keyboard to create  this number. Okay so this is my random number.  
You could get this from the ANU quantum vacuum  generator or from Cloudflare lava lamps or from  
your Intel chips RDRAND or, you know, in Python  land which pretty much always use a number 42,  
any of those are fine. So you pass in some  number, or you can pass in the current   tick count in nanoseconds. There's various  ways of getting some random starting point.  
And if we pass it into seed(), it's going to do  a bunch of modular divisions, and create a tuple  
of three things, and it's going to store them  in this global state. So rnd_state now contains  
three numbers. Okay, so, why did we do that? The  reason we did that is because now this function,  
Important tip on random numbers via process forking
which takes our random state, unpacks it  into three things and does again a bunch of  
modifications and moduloes, and then sticks  them together with various kind of weights.  
Modulo 1 (% 1.0), so this is how you can pull out  the decimal part. This returns random numbers.  
But the key thing I want you to understand is  that we pull out the random state at the start,  
we do some math thingies to it,  and then we store new random state.  
And so that means that each time  I call this, I'm going to get   a different number, right? So this is a random  number generator, and this is really important  
because lots of people in the Deep Learning  world screw this up —including me sometimes—  
which is to remember that random number generators rely on this state.
So let me show you where that will  get you if you're not careful.  
If we use a special thing called “fork()” that  creates a whole separate copy of this Python  
process. In one copy os.fork() returns True,  and the other in the other copy it returns False  
—roughly speaking—. So this copy here is this,  and if I say this, this version here, the True  
version, is the original non-clopied, it's called  the parent. And so in my else here this— so this  
will only be called by the parent, this will only  be called by the copy, which is called the child,   and each one I'm calling rand(). These  are two different random numbers, right?  
Wrong, they are the same number. Now, why is that?  That's because this process here and this process  
here are copies of each other and therefore, they  each contain the same numbers in random state.  
So, this is something that comes up in Deep  Learning all the time because in Deep Learning  
we often do parallel processing, for example,  to generate lots of augmented images at the  
same time, using multiple processes.  fastai used to have a bug, in fact,  
where we failed to correctly initialize the random  number generator separately, in each process.  
And in fact, to this day, at least as of  October 2022, torch.rand() itself, by default,  
fails to initialize the random number  generator. That's the same number. Okay,  
so, you've got to be careful. Now I have a  feeling NumPy gets it right, let's check. And  
is it how you do it —I don't  quite remember, we'll try—  
nope, okay, NumPy also doesn't.  How interesting. What about Python.  
And oh!, look at that! So Python does actually  
remember to re-initialize the  random stream at each fork.
So, you know, this is something that  like, even if you've experimented in   Python and you think everything's working  well in your data loader or whatever,  
and then you switch to Pytorch or NumPy and  now suddenly everything's broken. So, there,  
this is why we've spent some time re-implementing  random… the random number generator from scratch,  
partly because it's fun and interesting and partly  because it's important that you now understand  
that when you're calling rand, or any random  number generator, kind of the default versions  
in NumPy and Pytorch, this global state is going  to be copied, so you've got to be a bit careful.
Now, I will mention, our random number generator…  okay so this is, this is cool. %timeit,  
percent (%) is a special Jupiter or IPython  function and %timeit runs a piece of Python code  
this many times. So to call it 10 times, well  actually it'll do seven loops, and each one  
will be seven times and it'll take the mean and  standard deviation. So here I am going to generate  
random numbers, 7,840 times and put them into 10  
long chunks, and if I run that, it takes  me three milliseconds per loop. If I run it  
using Pytorch —this is the exact same thing in  Pytorch— it's going to take me 73 microseconds per  
loop. So as you can see, although we could use our  version, we're not going to because the Pytorch  
version is much, much faster. This is how we can  create a 784 by 10. And why would we want this?  
That's because this is a final layer of our neural  net, where if we're doing a linear classifier,   our linear weights we need to be  784 —because that's 28 by 28— by 10,  
because that's the number of possible  outputs, the number of possible digits.
All right, that is it. So, quite the intense  lesson, I think we can all agree. Should keep  
you busy for a week and thanks very much for  joining and see you next time. Bye everybody.

---

# 11

Hi everybody, welcome to Lesson 11. This is the third lesson in Part 2. 
Depending on how you count things, there's been  a lesson A and a lesson B, it's kind of the fifth   lesson in Part 2, I don't know what it is. So we'll just stick to calling it Lesson 11  
and avoid getting too confused. I'm already confused. 
My goodness, I've got so much stuff to show you. I'm only going to show you a tiny fraction of the   cool stuff that's been happening on the  forum this week, but it's been amazing. 
Showing student’s work
I'm going to start by sharing this beautiful  video from John Robinson —”Robinsn” I should say—,  
and I've never seen anything like this before. As you can see, it's very stable and it's really  
showing this beautiful movement between seasons. So what I did on the forum was I said to folks,  
hey, you should try interpolating  between prompts, which is what John did.  And I also said you should try using the last  image of the previous prompt interpolation  
as the initial image for the next prompt. And anyway, here it is, came out beautifully. 
John was the first to get that working,  so I was very excited about that. 
And the second one I wanted to show  you is this really amazing work from  
@sebderhy, Sebastian, who did something  that I've been thinking about as well. 
I'm really thrilled that he also thought about  this, which was he noticed that this update we do,  
unconditional embeddings plus guidance times  text embeddings minus unconditional embeddings  
[u + g * (t - u)], has a bit of a  problem, which is that it gets big. 
To show you what I mean by it gets big is like,  imagine that we've got a couple of vectors  
on this chart here. And so we've got, let's see, so we've got,  
let's just, okay, so we've got the original  unconditional piece here, so we've got u. 
So let's say this is u. Okay.  And then we add to that some amount of t minus u. So if we've got like t,  
let's say it's huge, right? And we've got u again. Then the difference between those is the vector  
which goes here, right? Now you can see here  that if there's a big difference between t and u,  
then the eventual update which actually  happens is, oopsie-daisy, I thought that  
was going to be an arrow. Let's try that again.  The eventual update which happens  
is far bigger than the original update. And so it jumps too far. 
So this idea is basically to say, well, let's  make it so that the update is no longer than the  
original unconditioned update would have been. And we're going to be talking more about norms  
later, but basically we scale  it by the ratio of the norms. 
And what happens is we start with  this astronaut and we move to  
this astronaut. And it's a subtle change, but you can   see there's a lot more before, after, before,  after, a lot more texture in the background. 
And like on the Earth, there's  a lot more detail before, after. 
You see that? And even little things  like before, the bridal kind of rains,   whatever, were pretty flimsy. Now they look quite proper. 
So it's made quite a big difference just  to kind of get this scaling correct. 
So there's a couple of other things that  Sebastian tried, which I'll explain in a  
moment, but you can see how some of them  actually resulted in changing the image. 
And this one's actually important because  the poor horse used to be missing a leg   and now it's not missing a leg, so that's good. And so here's the detailed one with its extra leg. 
So how did he do this? Well, so what he  did was he started with this unconditioned  
prompt plus the guidance times the difference  between the conditional and unconditioned. 
And then as we discussed, the next version,  well actually the next version we then saw is  
to basically just take that prediction and scale  it according to the difference in the lengths. 
So the norms is basically  the length of the vectors.  And so this is the second one I did in Lesson 9. You'll see it's gone from here. 
So when we go from 1a to 1b, you can see here it's  got, look at this, this boot's gone from nothing  
to having texture, this or whatever the hell this  thing is, suddenly he's got texture and look,  
we've now got proper stars in the sky. It's made a really big difference.  And then the second change is not just to rescale  the whole prediction, but to rescale the update. 
And when we rescale the update, it actually  not surprisingly changes the image entirely   because we're now changing the direction it goes. And so, I don't know, is this better than this?  
I mean, maybe, maybe not, but  you know, I think so, you know,   particularly because this was the difference that  added the correct fourth leg to the horse before. 
And then we can do both. We can rescale the   difference and then rescale the result. And then we get the best of both worlds. 
As you can see, big difference. We get a nice background.  This weird thing on his  back's actually become an arm. 
That's not what a foot looks like. That is what a foot looks like.  So these little details make a  big difference, as you can see. 
So this is a really cool, or  two really cool new things. 
New things tend to have wrinkles though. Wrinkle number one is after I shared on Twitter,  
Sebastian's approach, Ben Poole, who's a  Google Brain, I think, if I remember correctly,  
pointed out that this already exists. He thinks it's the same as what's   shown in this paper, which is a  diffusion model for text to speech. 
I haven't read the paper yet to check whether  it's got all the different options or whether   it's checked them all out like this. So maybe this is reinventing something  
that already existed and putting it into a  new field, which would still be interesting. 
Anyway, so hopefully, folks on the forum, you  can help figure out whether this paper's actually  
showing the same thing or not. And then the other interesting thing   was John Robinson got back in touch on the forum  and said, oh, actually, that tree video doesn't  
actually do what we think it does at all. There's a bug in his code.  And despite the bug, it  accidentally worked really well. 
So now we're in this interesting question of  trying to figure out like, oh, how did he create   such a beautiful video by mistake? OK, so reverse  engineering exactly what the bug did and then  
figuring out how to do that more intentionally. And this is great, right? It's really good to  
having a lot of people working on something. And the bugs often,   yeah, they tell us about new ideas. So that's very interesting. 
So watch this space where we find out what John  actually did and how come it worked so well. 
And then something that I just saw  like two hours ago on the forum,   which I'd never thought of before, but I'd  thought of something a little bit similar. 
Rekil Prashanth said, well, what if we took  this? So as you can see, all the students are  
really bouncing ideas of each other. It's like, oh, it's interesting.  We're doing different things  with a guidance scale. 
What if we take the guidance scale and rather than  keeping it at 7.5 all the time, let's reduce it. 
And this is a little bit similar to something I  suggested to Johno a few weeks ago where I said   —he was doing some stuff with like modifying  gradients based on additional loss functions. 
And I said to him, maybe you should just  use them like occasionally at the start,   because I think the key thing is once the  model kind of knows roughly what image  
it's trying to draw, even if it's noisy,  you know, you can let it do its thing.  And this is exactly what's happening  here is Rekil's idea is to say, well,  
let's decrease the guidance scale. So at the end, it's basically zero.  And so once it kind of is going in the  right direction, we let it do its thing. 
So this little doggy is with  the normal 7.5 guidance scale. 
Now have a look, for example, its eye here. It's pretty disin- uninteresting, pretty flat. 
And if I go to the next one, as you can see now,  actually look at the eye, that's a proper eye. 
Before, totally glassy black. Now proper eye.  Or like look at all this fur, very  textured, previously very out of focus. 
So this is, again, a new technique. So I love this, you know, you folks are  
trying things out, and some things are  working and some things not working.  And that's all good. I kind of feel like you're  
going to have to slow down because I'm  having trouble keeping up with you all.  But apart from that, this is great. Good work. 
I also wanted to mention on a different  theme to check out Alex's notes  
on the lesson, because I thought  he's done a fantastic job of showing   like how to study, how to study a lesson. And so what Alex did, for example, was he made  
a list in his notes of all the different steps  we did as we started the From the Foundations. 
What is the library that it comes  from, links to the documentation. 
And I know that Alex's background actually  is history, you know, not computer science. 
And so, you know, for somebody moving into a  different field like this, this is a great idea,   you know, particularly to be able to like look  at like, okay, what are all the things that  
I'm going to have to learn and read about? And  then he did something which we always recommend,  
which is to try the lesson on a new data set. And he very sensibly picked out the Fashion MNIST  
data set, which is something we'll be using a lot  in this course, because it's a lot like MNIST. 
And it's just different enough to be interesting. And so he described in his post or his notes,  
how he went about doing that. And then something else I thought   was interesting in his notes at the very  end was he just jotted down my tips. 
It's very easy when I throw a tip out  there to think, oh, that's interesting,  
that's good to know. And then it can disappear.  So here's a good way to make sure you  don't forget about all the little tricks. 
And I think I've put those notes in the  forum wiki, so you can check them out if  
you'd like to learn from them as well. So I think this is a great role model.  Good job, Alex. Okay, so  
during the week, Johno taught us about a new paper  that had just come out called DiffEdit, and he  
told us he thought this was an interesting paper. And it came out during the week, and I thought it  
might be good practice for us to  try reading this paper together. 
So let's do that. So here's the paper, DiffEdit.  And you'll find that  
Workflow on reading an academic paper
probably the majority of papers that you come  across in deep learning will take you to arXiv. 
arXiv is a preprint server. So these are models, these  
are papers that have not been peer reviewed. I would say in our field, we don't generally  
or I certainly don't generally care about that  at all, because we have code, we can try it,  
we can see things, whether it works or not. You know, we tend to be very, you know, most   papers are very transparent about here's what we  did and how we did it, and you can replicate it. 
And it gets a huge amount  of peer review on Twitter.  So if there's a problem, generally within  24 hours, somebody has pointed it out. 
So we use arXiv a lot. And if you wait until it's   been peer reviewed, you know, you'll be way out  of date because this field is moving so quickly. 
So here it is on arXiv, and we can  read it by clicking on the PDF button. 
I don't do that. Instead I click on this little button up here,  
which is the save to Zotero button. So I figured I'd show you   like my preferred workflows. You don't have to do the same thing. 
There are different workflows, but here's one that  I find works very well, which is Zotero is a piece  
of free software that you can download for Mac,  Windows, Linux, and install a Chrome connector. 
Oh, Tanishq is saying the button's covered. All right.  So in my taskbar, I have a button that  I can click that says, save to Zotero,  
sorry, not taskbar, Chrome menu bar. And when I click it, I'll show you what happens.  So after I've downloaded this, the paper will  automatically appear here in this software,  
which is Zotero. And so here it is, DiffEdit. 
And you can see it's told us, it's got here,  the abstract, the authors, where it came from. 
And so later on, I can go and like, if I want to  check some detail, I can go back and see the URL.  I can click on it, pops up. And so in this case, what I'm going to  
do is I'm going to double click on it. And that brings up the paper. 
Now the reason I like to read my papers  in Zotero is that I can annotate them,  
edit them, tag them, put them in folders and  so forth, and also add them to my kind of  
reading list directly from my web browser. So as you can see, I've started this fast   diffusion folder, which is actually a  group library, which I share with the  
other folks working on this fast diffusion  project that we're all doing together.  And so we can all see the same paper library. So @mariboo on YouTube chat is asking,  
is this better than Mendeley? Yeah, I used to  use Mendeley and it's kind of gone downhill. 
I think Zotero is far, far better,  but they're both very similar. 
Okay, so we double click on it. It opens up and here is a paper. 
Read DiffEdit paper
So reading a paper is always  extremely intimidating. 
And so you just have to do it anyway. And you have to realize that your goal   is not to understand every word. Your goal is to understand  
the basic idea well enough that, for  example, when you look at the code,   hopefully it comes with code, most things do,  that you'll be able to kind of see how the code  
matches to it and that you could try writing  your own code to implement parts of it yourself.  So over on the left, you can  open up the sidebar here. 
So I generally open up the table of  contents and get a bit of a sense of, okay,  
so there's some experimental results. There's some theoretical results. 
Introduction related work, okay, tells us about  this new DiffEdit thing, some experiments. 
Okay. So it's a pretty standard   approach that you would see in papers. So I would always start with the abstract. 
Okay, so what's it saying this does? So generally  it's going to be some background sentence or two  
about how interesting this field is. It's just saying, well, image   generation is cool, which is fine. And then they're going to tell us  
what they're going to do, which is they're  going to create something called DiffEdit. 
And so this is a, what is it for? It's going  to use text condition diffusion models.  So we know what those are now. That's what we've been using. 
That's where we type in some text and get  back an image of that, that matches the text,  
but this is going to be different. It's the task of semantic image editing.  Okay. We don't know what that is yet. 
So let's put that aside and think, okay,  let's make sure we understand that later.  The goal is to edit an  image based on a text query. 
Oh, okay. So we're going to edit an image based on text.  How on earth would you do that? Ah, they're  going to tell us right away what this is. 
Semantic image editing. It's an extension of image generation   with an additional constraint, which  is the generated image should be as  
similar as possible to the given input. And so generally, as they've done here,   there's going to be a picture  that shows us what's going on. 
And so in this picture, you can see  here, an example, here's an input image. 
And originally it was attached  to a caption, a bowl of fruits. 
Okay. So we want to change this into a bowl of pears.  So we type a bowl of pears and it generates,  Oh, a bowl of pears, or we could change it  
from a bowl of fruit to a basket of fruits  and, Oh, it's become a basket of fruits.  Okay. So I think I get the idea, right?  
What it's saying is that we can edit an image  by typing what we want that image to represent. 
So this actually looks a lot like the  paper that we looked at last week. 
So that's cool.  So the abstract says that currently, so I  guess there are current ways of doing this,  
but they require you to provide a mask. That means you have to basically draw the   area you're replacing. Okay. 
So that sounds really annoying,  but our main contribution.  So what this paper does is we  automatically generate the mask. 
So they simply just type in the  new query and get the new image.  So that sounds actually really impressive. So if you read the abstract and you think,  
I don't care about doing that, then you can skip  the paper, you know, or, or look at the results  
and if the results don't look  impressive, then just skip the paper.  So that's, that's kind of your first point  where we can be like, okay, we're done. 
But in this case, this sounds great.  The results look amazing. So I think we should keep going. 
Okay. “achieves state-of-the-art   editing performance, of course. Fine.  Let me try some, right, whatever. Okay. 
So the introduction to a paper is  going to try to give you a sense of,  
you know, what they're trying to do. And so this first paragraph here is  
just repeating what we've already read in the  abstract and repeating what we see in figure one.  So saying that we can take a text query,  like a basket of fruits, see the examples. 
All right, fine. We'll skip through there.  So key thing about academic papers  is that they are full of citations. 
You should not expect to read  all of them because if you do,   then to read each of those citations, that's full  of citations and then they're full of citations. 
And before you know it, you've read  the entire academic literature,   which has taken you 5,000 years. So for now, let's just recognize  
that it says text conditional image  generations undergoing revolution.  Here's some examples. Well, fine.  We actually already know that. Okay. 
DALL-E is cool. Latent Diffusion.  That's what we've been using. That's cool.  Imagen, apparently that's cool. So cool. 
All right. So we kind of know that.  So generally there's this like, okay, our  area that we're working on is important.  And in this case, we already agree it's important  so we can skip through it pretty quickly. 
There're vast, vast amounts of data are used. Yes, we know. 
Okay. So diffusion models are interesting.  Yes, we know that they de-noise  starting from Gaussian noise. 
We know that. So you can see like   there's a lot of stuff once you kind of in  the field, you can skip over pretty quickly. 
You can guide it using CLIP guidance. Yeah.  That's what we've been doing. We know about that.  Oh, wait, this is new. Or by inpainting,  
by copy pasting pixel values outside a mask. All right.  So there's a new technique that we haven't  done, but I think it makes a lot of intuitive  
sense that is during that diffusion process. If there are some pixels, you don't want to change  
such as all the ones that aren't orange here, you  can just paste them from the original after each  
stage of the diffusion. All right.  That makes perfect sense. If I want to know more about that,   I could always look at this paper,  but I don't think I do for now. 
Okay. And again,   it's just repeating something  they've already told us that they're   they require us to provide a mask. So that's a bit of a problem. 
And then, you know, this is interesting. It's also says that when you mask out an area  
that's a problem because if you're trying  to, for example, change a dog into a cat,   you want to keep the animal's color and pose. So this is a new technique, which is not deleting  
the original, not deleting a section  and replacing it with something else,   but it's actually going to take advantage of  knowledge about what that thing looked like. 
So that, this is two cool new things. So hopefully at this point, we know  
what they're trying to achieve. If you don't know what they're   trying to achieve when you're reading a  paper, the paper won't make any sense. 
So again, that's a point where you should stop. Maybe this is not the right   time to be reading this paper. Maybe you need to read some of the references. 
Maybe you need to look more at the examples so  you can always skip straight to the experiments. 
So I often skip straight to the experiments. In this case, I don't need to because they've put  
enough experiments on the very first  page for me to see what it's doing.  So yeah, don't always read it from top to bottom. Okay, so. 
All right, so they've got some examples of  conditioning a diffusion model on an input   without a mask. Okay. 
For example, you can use a noise version  of the input as a starting point.  Hey, we've done that too. So as you can see, we've already covered a lot  
of the techniques that they're referring to here. Something we haven't done, but makes a lot of  
sense is that we can look at the distance  to the input image as a loss function.  Okay. So that makes  
sense to me and there's some references here.  All right. So we're going to create  
this new thing called DiffEdit. It's going to be amazing.  Wait till you check it out. Okay, fine. 
Okay. So that's the introduction.  Hopefully you found that useful to  understand what we're trying to do. 
The next section is generally  called related work as it is here,   and that's going to tell  us about other approaches. 
So if you're doing a deep dive, this  is a good thing to study carefully. 
I don't think we're going  to do a deep dive right now.  So I think we can happily skip over it. We could kind of do a quick glance  
of like, oh, image editing, conclude  colorization, retouching style transfer. 
Okay, cool. Lots of interesting topics.  Definitely getting more excited  about this idea of image editing. 
And there's some different techniques.  You can use CLIP guidance. Okay. 
They can be computationally expensive. We can use diffusion for image editing. 
Okay, fine. We can use CLIP to help us.  So there's a lot of repetition in  these papers as well, which is nice  
because we can skip over it pretty quickly. More about the high computational costs. 
Okay. So they're saying this is going   to be not so computationally expensive. That sounds hopeful. 
And often the very end of the related  work is most interesting as it is here   where they've talked about how somebody  else has done “concurrent to ours”. 
Somebody else is working at exactly the same time  and they've looked at some different approach. 
Okay. So not sure we   learned too much from the related work, but  if you were trying to really do the very,  
very best possible thing, you could study the  related work and get the best ideas from each. 
Okay. Now, background.  So this is where it starts to look scary. I think we could all agree. 
Understanding the equations in the “Background” section
And this is often the  scariest bit, the background.  This is basically saying like  mathematically, here's how the  
problem that we're trying to solve is set up. And so we're going to start by looking at   denoising diffusion probabilistic models, DDPM. Now, if you've watched Lesson 9B with Wasim  
and Tanishq, then you've already  seen some of the math of DDPM. 
And the important thing to recognize is that  basically no one in the world pretty much is  
going to look at these paragraphs of text  and these equations and go, oh, I get it. 
That's what DDPM is. That's not how it works, right?  
To understand DDPM, you would have to read  and study the original paper, and then you  
would have to read and study the papers it's  based on and talk to lots of people and watch  
videos and go to classes just like this one. And after a while, you'll understand DDPM. 
And then you'll be able to look at  this section and say, oh, okay, I see. 
They're just talking about this  thing I'm already familiar with.  So this is meant to be a reminder  of something that you already know. 
It's not something you should  expect to learn from scratch.  So let me take you through  these equations somewhat briefly  
because Wasim and Tanishq have kind of done them  already, because every diffusion paper pretty much  
is going to have these equations. Okay.  So, oh, and I'm just going to read something  that Johno's pointed out in the chat. 
He says, it's worth remembering the background  is often written last and tries to look smart   for the reviewers, which is correct. So feel free to read it last too. 
Yeah, absolutely. I think the main reason to read it is to  
find out what the different letters  mean, what the different symbols mean,   because they'll probably refer to them later. But in this case, I want to actually take  
this as a way to learn how to read math. So let's start with this very first equation,  
which, how on earth do you even read this? So the  first thing I'll say is that this is not an e,  
right? It's a weird looking e. And the reason it's  a weird looking e is because it's a Greek letter.  And so something I always recommend  to students is that you learn  
the Greek alphabet because it's much easier  to be able to actually read this to yourself. 
So here's another one, right? If  you don't know, that's called theta.  I guess you have to read it as  like circle with line through it. 
It's just going to get confusing trying to read  an equation where you just can't actually say  
it out loud. So  
what I suggest is that you learn that, learn the  Greek alphabet and let me find the right place. 
So it's very easy to look it up just  on Wikipedia is the Greek alphabet. 
But if we go down here, you'll see  they've all got names and we can go and   try and find our one curvy e. Okay, here it is. Epsilon and oh, circle with a line through it. 
Theta. All right.  So practice and you will get used to  recognizing these. So you've got epsilon theta. 
This is just a weird curly L. So that's,  this is used for the loss function. 
Okay. So how do we find out what this   symbol means and what this symbol means? Well,  what we can do is there's a few ways to do it. 
One way, which is kind of cool is we can  use a program called MathPix which, MathPix. 
Here we are, MathPix.  And what it does is you basically select anything  on your screen and it will turn it into LaTeX. 
So that's one way you can do this is you can  select on the screen, it turns it into LaTeX.  And the reason it's good to turn it into  LaTeX is because LaTeX is written as actual  
stuff that you can search for on Google. So that's technique number one. 
Technique number two is you can  download the other formats of the paper  
and that'll have a download source. And if we say download source,  
then what we'll be able to do is we'll be to  actually open up that LaTeX and have a look at it. 
So we'll wait for that to  download while that's happening.  Let's keep moving along here. So in this case, we've got these two bars. 
So can we find out what that means?  So we could try a few things. 
We could try looking for two  bars (“||”), maybe math notation. 
Oh, here we are. Looks hopeful.  What does this mean in mathematics? Oh, and  here there's a glossary of mathematical symbols. 
Here there's a meaning of this in math. 
So that looks hopeful. Okay.  So it definitely doesn't look like this. It's not between two sets of letters. 
Ah, but it is around something that looks hopeful. So it looks like we found it. 
It's a vector norm. Okay.  So then you can start looking for these things up. So we can say norm or maybe vector norm. 
And so once you can actually find the term,  then we kind of know what to look for. 
Okay. So  
in our case, we've got this  surrounding all this stuff. 
And then there's twos here and here. What's going on here? All right. 
If we scroll through. Oh, this is pretty close actually. 
So okay.  So two bars can mean a matrix norm,  otherwise a single for a vector norm. 
That's just here in particular. So it looks like we don't have to worry   too much about whether it's one or two bars. Oh, and here's the definition. 
Oh, that's handy. So we've got the two one.  All right. So it's equal to root sum of squares. 
So that's good to know. So this norm thing means a root sum of squares. 
But then we've got a two up here. Well, that just means squared.  Ah, so this is a root sum of squares squared. 
Well, the square of a square  root is just the thing itself.  Ah, so actually this whole thing  is just the sum of squares. 
It's a bit of a weird way to write it in a sense. You could perfectly well have just  
written it as, you know, like sum of,  you know, whatever it is, squared. 
Fine. But there we go. 
Okay. And then   what about this thing here? Weird E thing. So how would you find out what the weird  
E thing is? Okay, so our LaTeX  has finally finished downloading. 
And if we open it up, we can find  there's a dot tex file in here.  Here we are, main.tex. So we'll open it. 
And it's not the most, you know, amazingly  smooth process, but you know, what we could  
just do is we could say, okay, it's just after  it says, minimizing the de-noising objective.  Okay. 
So let's search for minimizing the de- oh, here  it is, minimizing the de-noising objective. 
So the LaTeX here, let's get it  both on the screen at the same time. 
Okay. So here it is.  L mathcal{L} = mathbb{E} x naught t epsilon. Okay. 
And here's that vertical bar thing, epsilon minus  epsilon theta x_t, and then the bar thing two two. 
All right. So the thing that we've got new is mathbb{E}.  Okay. So finally, we've got something we can search for. 
mathbb{E}. Ah, fantastic.  What does mathbb{E} mean? That's  the expected value operator. 
Aha, fantastic. All right.  So it takes a bit of fussing around, but once  you've got either Mathpix working or actually  
another thing you could try, because Mathpix is  ridiculously expensive in my opinion, is there is  
a free version called pix2tex  that actually is a Python thing. 
And you could actually even have fun playing with  this because the whole thing is just a PyTorch  
Python script. And it even describes,   you know, how if you used a transformers model and  you can train it yourself in Colab and so forth. 
But basically, as you can see, yeah, you can snip  and convert to LaTeX, which is pretty awesome. 
So you could use this instead  of paying the Mathpix guys. 
Anyway, so we are on the right track now, I think. So expected value. 
And then we can start reading about what  expected value is, and you might actually   remember that because we did a bit of it in  high school, at least in Australia we did. 
It's basically like, let's maybe jump over here. 
So expected value of something is saying, what's  the likely value of that thing? So for example,  
let's say you toss a coin, which  could be heads or it could be tails.  And you want to know how often it's heads. And so maybe we'll call heads one, tails zero. 
So you toss it and you get a one, zero,  zero, one, one, zero, one, zero, one. 
Okay. And so forth.  Right. And then you can calculate the mean of that. 
Right. So if that's X, you can calculate X bar   the mean, which would be the sum of all  that divided by the count of all that. 
So it'd be one, two, three, four, five, five  divided by one, two, three, four, five, six,  
seven, eight, nine. Okay.  So that would be the mean. But the expected value is like, well,  
what do you expect to happen? And we can calculate  that by adding up for all of the possibilities  
for each, I don't know what it's  called them, X, for each possibility X.  How likely is X and what score do you get if you  get X? So in this example of heads and tails,  
our two possibilities is that we  either get heads or we get tails.  So if for the version where X is  heads, we get probability is 0.5  
and the score, if it's an X, I guess I  should use that, the score if it's an X  
is going to be one. And then what about   tails? The tails, the probability is 0.5  and the score, if you get tails is zero. 
And so overall the expected is  0.5 times one plus zero is 0.5. 
So our expected score, if we're tossing  a coin is 0.5, if getting heads is a win. 
Let me give you another example. Another example is let's say that   we're rolling a die and we want to know  what the expected score is if we roll a die. 
So again, we could roll it a bunch  of times and see what happens. 
And so we could sum all that up, (as) before   and divide it by the count. And that'll tell us the mean  
for this particular example, but what's the  expected value more generally? Well, again,  
it's the sum of all the possibilities of the  probability of each possibility times that score. 
So the possibilities for rolling  a die is that you can get a one,  
a two, a three, a four, a five or a six. The probability of each one is a sixth. 
Okay. And the score that you get is,   well, it's this, this is the score. And so then you can multiply all these  
together and sum them up, which would be one  sixth plus two sixths plus three sixths plus  
four sixths plus five sixths plus six sixths. And that would give you the expected value of  
that particular thing, which is  rolling die, rolling, rolling a die. 
So that's what expected value means. All right.  So that's a really important concept that's  going to come up a lot as we read papers. 
And so in particular, this is telling  us what are all the things that we're   averaging it over that was the expectations over. And so there's a whole lot of letters here. 
You're not expected to just know what they are.  In fact, in every paper, they could  mean totally different things.  So you have to look immediately  underneath where they'll be defined. 
So X0 is an image. It's an input image. 
Epsilon is the noise and the noise has a mean of 0  and a standard deviation of I, which if you watch  
the Lesson 9B you'll know it's like a standard  deviation of one when you're doing multiple  
normal variables. Okay.  And then this is kind of confusing. Eta just on its own is a normally  
distributed random variables. It's just grabbing random numbers,   but Eta-the… sorry, psilon, but  Epsilon-heta is a noise estimator. 
That means it's a function. You can tell it's a function kind of, cause it's   got these parentheses and stuff right next to it. So that's a function. 
So presumably most functions like this  in these papers are neural networks. 
Okay. So we're finally at a point where this   actually is going to make perfect sense. We've got the noise. 
We've got the prediction of that noise. We subtract one from the other. 
We square it and we take the expected value. So in other words, this is mean squared error. 
So wow, that's a lot of fiddling  around to find out that we've,   this whole thing here means mean squared error. So the loss function is the mean squared error. 
And unfortunately I don't think the paper  ever says that it says minimizing the   de-noising objective L bladi bladi bladi. But anyway, we got there eventually. 
Fine. We also, as well as   learning about X0, we also learn here about Xt. And so Xt is the original unnoised image  
times some number plus some noise  times one minus that number. 
Okay. And so   hopefully you'll recognize this from Lesson 9B. This is the thing where we reduce the value of  
each pixel and we add noise to each pixel. So that's that. 
All right. So I'm not going to keep going through it,   but you can kind of basically get the idea here  is that once you know what you're looking for,  
the equations do actually make sense. But all this is doing is, remember,  
this is background, right? This is  telling you what already exists.  So this is telling you, this is what a DDPM is. And then it tells you what a DDIM is. 
DDIM is, look, just think of it  as a more recent version of DDPM.  It's some very minor changes to the way  it's set up, which allows us to go faster. 
Okay. So the thing is though,   once we keep reading, what you'll find is  none of this background actually matters. 
But, you know, I thought we'd kind of go through  it just to get a sense of like, what's in a paper. 
Okay.  So for the purpose of our background,  it's enough to know that DDPM and DDIM  
are kind of the foundational papers on  which diffusion models today are based. 
Okay. 
So the encoding process, which encodes  an image onto a latent variable. 
Okay. And then   this is basically adding noise. This is called DDIM encoding. 
And the thing that goes from the input image to  the noise image, they're going to call capital Er. 
And r is the encoding ratios.  That's going to be something like  how much noise are we adding. 
If you use small steps, then decoding that. So going backwards gives you back the original  
image. Okay.  So that's all the stuff that we've learned about. That's what diffusion models are. 
All right. So this looks like a very useful picture. 
3 steps of DiffEdit
So maybe let's take a look and see what this says. So what is DiffEdit? DiffEdit has three steps. 
Step one. We add noise to the input image.  That sounds pretty normal. Here's our input image X0. 
Okay. And we add noise to it.  Fine. And then we denoise it. 
Okay. Fine.  Ah, but we denoise it twice. One time we denoise it  
using the reference text R horse. Well this special symbol  
here means nothing at all. So either unconditional or horse.  All right. So we do it once using the word horse. 
So we take this and we decode it, estimate the  noise, and then we can remove that noise on the  
assumption that it's a horse. Then we do it again. 
The second time we do that noise, when  we calculate the noise, we pass in  
our query Q, which is zebra. Wow.  Those are going to be very different noises. The noise for horse is just going to  
be literally these Gaussian pixels. These are all dots, right? Because it is a horse.  But if the claim is no, no, this is  actually a zebra, then all of these  
pixels here are all wrong. They're all the wrong color.  So the noise that's calculated, if we say this  is our query, it's going to be totally different  
to the noise if we say this is our query. And so then we just take one minus the other,  
and here it is here. So “we derive a mask based on   the difference in the denoising results”. And then you take that and binarize it. 
So basically turn that into ones and zeros. So that's actually the key idea. 
That's a really cool idea, which is that once  you have a diffusion model that's trained,  
you can do inference on it where you tell it the  truth about what the thing is, and then you can  
do it again, but lie about what the thing is. And in your lying version, it's going to say,  
okay, all the stuff that doesn't  match zebra must be noise.  And so the difference between the noise prediction  when you say, hey, it's a zebra versus the noise  
prediction when you say, hey, it's a horse  will be all the pixels that it says, no,  
these pixels are not zebra. The rest of it, it's fine.  There's nothing particularly about the  background that wouldn't work with a zebra. 
Okay, so that's step one. So then step two is we take  
the horse and we add noise to it. Okay, that's this Xr thing  
that we learned about before. And then step three, we do “decoding  
conditioned on the text query” using the mask  to replace the background with pixel values. 
So this is like the idea that we heard about  before, which is that during the inference time,  
as you do diffusion from this fuzzy horse,  what happens is that we do a step of diffusion  
inference, and then all these black pixels, we  replace with the noise version of the original. 
And so we do that multiple times. And so that means that the original pixels  
in this black area won't get changed. And that's why you can see in this  
picture here, and this picture  here, the backgrounds all the same.  And the only thing that's changed is that  the horse has been turned into a zebra. 
So this paragraph describes it. And then you can see here,   it gives you in a lot more detail. And the detail often has all kinds  
of like little tips about things they tried  and things they found, which is pretty cool. 
So I won't read through all that because it  says the same as what I've already just said.  One of the interesting little things they note  here actually is that this binarized mask,  
so this difference between the R decoding and  the Q decoding tends to be a bit bigger than  
the actual area where the horse is, which you  can kind of see with these legs, for example.  And their point is that they  actually say that's a good thing  
because actually often you want to slightly  change some of the details around the object. 
So this is actually fine. All right.  So we have a description of what the thing is. Lots of details there. 
And then here's the bit that I totally skip, the  bit called theoretical analysis, where this is  
the stuff that people really generally just  add to try to get their papers past review.  You have to have fancy math. And so they're basically proving,  
you can see what it says here, insight  into why this component yields better  
editing results than other approaches. I'm not sure we particularly care because like  
it makes perfect sense what they're doing. It's intuitive and we can see it works.  I don't feel like I need it  proven to me, so I skip over that. 
Homework
So then they'll show us their experiments to tell  us what data sets they did the experiments on. 
And so then, you know, they have  metrics with names like LPIPS and CSFID. 
You'll come across FID a lot. This is just a version of that.  We're basically, they're trying to  score how good their generated images. 
We don't normally care about that either. They care because they need to be able to say,   you should publish our paper because it  has a higher number than the other people  
that have worked on this area. In our case, we can just say,  
you know, it looks good. I like it. 
So excellent question in the chat from  Mikołaj, which is, so with this only work   on things that are relatively similar. And I think this is a great point,  
right? This is where understanding this helps  to know what its limitations are going to be. 
And that's exactly right. If you can't come up with a   mask for the change you want, this isn't  going to work very well on the whole. 
Yeah. Because those,   the masked areas, the pixel is going to be copied. So for example, if you wanted to change it from,  
you know, a bowl of fruits to a bowl of  fruits with a bokeh background, or like  
a bowl of fruits with, you know, with a purple  tinged photo of a bowl of fruit, if you want the  
whole color to change, that's not going to work,  right? Because you're not masking off an area.  Yeah. So by understanding the  
detail here, Mikołaj has correctly recognized  a limitation or like, what's this for? This is  
for things where you can just say, just change  this bit and leave everything else the same. 
All right. So there's lots   of experiments. So yeah.  For some things that you care about the  experiments a lot, if it's something like  
classification for stuff for generation,  the main thing you probably want to look at   is the actual results. And so,  
and often for whatever reason, I guess, because  this is most people read these electronically,   the results often you have to zoom into a lot  to be able to see whether they're really good. 
So here's the input image. They want to turn this into an English Foxhound. 
So here's the thing they're comparing themselves  to SDEdit and change the composition quite a lot  
and their version, it hasn't changed it at all.  It's only changed the dog. And  ditto here, semi-trailer truck,  
SDEdits totally changed it and DifFdit hasn't. So you can kind of get a sense of like, you know,  
the authors showing off what they're good at here. This is what this technique is effective at doing,  
changing animals and vehicles and so forth. It does a very good job of it. 
All right.  So then there's going to be a conclusion  at the end, which I find almost never adds  
anything on top of what we've already read. And as you can see, it's very short anyway. 
Now quite often the appendices  are really interesting.  So don't skip over them. Often you'll find like more examples of pictures. 
They might show you some examples of pictures  that didn't work very well, stuff like that.  So it's often well worth  looking at the appendices. 
Often some of the most  interesting examples are there.  And that's it. So that is,  
I guess our first full on paper walkthrough. And it's important to remember this is not like  
a carefully chosen paper that we've picked  specifically because you can handle it. 
Like this is the most interesting  paper that came out this week.  And so, you know, it gives you a  sense of what it's really like. 
And for those of you who are ready to try  something that's going to stretch you,  
see if you can implement any of this paper. So there are three steps. 
The first step is kind of the most  interesting one, which is to generate,   automatically generate a mask. 
And the information that you have and  the code that's in the Lesson 9 notebook  
actually contains everything you need to do it. So maybe give it a go.  Maybe if you can mask out the area of a  horse that does not look like a zebra. 
And that's actually useful in itself. Like that allows you to create segmentation  
masks automatically. So that's pretty cool.  And then if you get that working, then  you can go and try and do step two. 
If you get that working, you  can try and do step three.  And this only came out this week. So I haven't really seen  
examples of easy to use interfaces to this.  So here's an example of a paper that you could be  the first person to create a call interface to it. 
So there's some, yeah,  there's a fun little project.  And even if you're watching this a  long time after this was released  
and everybody's been doing this for years, still  good homework, I think, to practice if you can. 
All right, I think now's a good  time to have a 10 minute break. 
So I'll see you all back here in 10 minutes. 
Okay, welcome back. One thing during the   break that Diego reminded us about, which I  normally describe and I totally forgot about  
this time is Detexify, which is another really  great way to find symbols you don't know about. 
So let's try it for that expectation. So if you've got a Detexify and you draw the  
thing, it doesn't always work fantastically  well, but sometimes it works very nicely. 
Yeah, in this case, not quite. What about the double line thing?  
It's good to know all the techniques, I guess. 
I think it could do this one. I guess part of the problem is there's   so many options that actually, you know, okay,  in this case, it wasn't particularly helpful. 
Normally it's more helpful than that. I mean, if we use a simple one like   Epsilon, I think it should be fine. 
There's a lot of room to improve this app  actually, if anybody's interested in a project.  I think you could make it,  you know, more successful. 
Okay, there you go. Sigma sum, that's cool.  Anyway, so it's another useful thing  to know about just Google for Detexify. 
Okay, so let's move on with  our From the Foundations now. 
And so we were working on trying to  at least get the start of a forward  
pass of a linear model or a simple  multi-layer perceptron for MNIST going. 
And we had successfully created a basic tensor. We've got some random numbers going. 
Matrix multiplication from scratch
So what we now need to do is we now need to  be able to multiply these things together,  
matrix multiplication. So matrix multiplication,  
to remind you, in this case, so we're  doing MNIST, right? So we've got,  
I think we're going to use a subset. Let's see.  Yeah. Okay.  So we're going to create a matrix called  m1, which is just the first five digits. 
So m1 will be the first five digits. 
So five rows and dot, dot, dot; dot, dot, dot. and then 780, what was it again? 784 columns,  
784 columns, because it's 28 by  28 pixels and we flattened it out. 
So this is our first matrix  and our matrix multiplication.  And then we're going to  multiply that by some weights. 
So the weights are going to  be 784 by 10 random numbers. 
So for every one of these 784 pixels,  each one is going to have a weight. 
So 784 down here, 784 by 10. So this first column, for example,  
is going to tell us all the weights in  order to figure out if something's a zero. 
And the second column will have  all the weights in deciding the   probability of something's a one and so forth. That's assuming we're just doing a linear model. 
And so then we're going to multiply  these two matrices together.  So when we multiply matrices  together, we take row one  
of matrix one, and we take column one of  matrix two, and we take each one in turn. 
So we take this one and we take  this one, we multiply them together. 
And then we take this one and this  one, and we multiply them together. 
And we do that for every element wise pair,   and then we add them all up. And that would give us the value  
for the very first cell. That would go in here.  That's what matrix multiplication is. Okay, so  
let's go ahead then and create our random  numbers for the weights, since we're   allowed to use random number generators now. And for the bias, we'll just use a bunch of  
zeros to start with. So the bias is just   what we're going to add to each one. And so for our matrix multiplication,  
we're going to be doing a little mini batch here. We're going to be doing five rows of,  
as we discussed, five rows of,  so five images flattened out. 
And then multiply by this weights matrix. 
So here are the shapes. m1 is five by 784, as we saw.  m2 is 784 by 10. Okay, so keep those in mind. 
So here's a handy thing, m1.shape contains  two numbers and I want to pull them out. 
I want to call the, I'm going to think of that  as, I'm going to actually think of this as like  
A and B rather than m1 and m2. So this is like A and B.   So the number of rows in A and the number  of columns in A, if I say equals m1.shape,  
that will put 5 in ar and 784 in ac. So you'll probably notice this. 
I do this a lot, this de-structuring. We talked about it last week too.  So we can do the same for m2.shape, put  that into B rows (br) and B columns (bc). 
And so now if I write out ar, ac and br, bc, you  can again see the same things from the sizes. 
So that's a good way to kind of give  us the stuff we have to loop through.  So here's our results. So our resultant tensor,  
well, we're multiplying together all  of these 784 things and adding them up. 
So the resultant tensor is going to be 5 by 10. 
And then each thing in here is the result  of multiplying and adding 784 pairs. 
So the result here is going to start  with zeros and this is the result. 
And it's going to contain ar rows,  five rows and bc columns, 10 columns,  
5 comma 10. Okay.  So we have to fill that in. And so to do a matrix multiplication,  
we have to first, we have to go  through each row one at a time. 
And here we have that: go  through each row, one at a time,  
and then go through each column, one at a time. And then we have to go through each pair  
in that row column one at a time. So there's going to be a loop in a loop in a loop. 
So here's going to loop over each row. And here we're going to loop over each column. 
And then here we're going to loop —so each  column bc. And then here we're going to loop over   each column of a, which is going to be the same  as the number of rows of b, which we can see here,  
ac: 784, br: 784, they're the same. So it wouldn't matter whether we said ac or br. 
So then our result for that row and that column,  we have to add onto it the product of i, k in the  
first matrix, by k, j in the second matrix. So k is going up through those 784. 
And so we're going to go across the columns and  down, sorry, across the rows and down the columns. 
So across the row, well, as  it goes down this column. 
So here is the world's most naive, slow,  uninteresting matrix multiplication. 
And if we run it, okay, it's done something. We have successfully, apparently, hopefully  
successfully, multiplied the matrices m1 and m2. It's hard to read this, I find, because  
punch cards used to be 80 columns wide. We still assume screens are 80 columns wide.  Everything defaults to 80 wide, which is  ridiculous, but you can easily change it. 
So if you say set_print_options,  you can choose your own line width. 
You can see, well, we know that's 5 by 10. We did it before.  So if we change the line width,  okay, that's much easier to read now. 
You can see here are the 5 rows and here are  the 10 columns for that matrix multiplication. 
I tend to always put this  at the top of my notebooks   and you can do the same thing for NumPy as well. 
So what I like to do, this is really important,  is when I'm working on code, particularly numeric  
code, I like to do it all step by step in Jupyter. And then what I do is once I've got it working is  
I copy all the cells that have implemented  that and I paste them and then I select them  
all and I hit shift+M to merge, get rid of  anything that prints out stuff I don't need. 
And then I put a header on the  top, give it a function name,  
and then I select the whole lot and I hit  control or Apple right square bracket and  
I've turned it into a function. But I still keep the stuff above   it so I can see all the step by step  stuff for learning about it later. 
And so that's what I've done  here to create this function. 
And so this function does exactly  the same things we just did.  And we can see how long it  takes to run by using %time. 
And it took about half a second,  which gosh, that's a long time  
to generate such a small matrix. This is just to do five MNIST digits. 
So that's not going to be great. We're going to have to speed that up. 
I'm actually quite surprised at how slow  that is because there's only 39,200. 
So if you look at how we've got a loop within  a loop within a loop, a loop within a loop  
within a loop, it's doing 39,200 of these. So Python, yeah, Python, when you're just  
doing Python, it is slow. So we can't do that.  That's why we can't just write Python. But there is something that kind  
Speed improvement with Numba library
of lets us write Python. We could instead use Numba. 
Numba is a system that takes Python and  turns it into basically into machine code. 
And it's amazingly easy to do. You can basically take a function  
and write njit, @njit on top. And what it's going to do is it's going to  
look at —the first time you call this function,  it's going to compile it down to machine code  
and it will run much more quickly. So what I've done here is I've taken the  
innermost loop. So just looping through and adding up all these. 
So start at zero, go through and add up all  those just for two vectors and return it. 
This is called a dot product in linear algebra. So we'll call it dot. 
And so Numba only works with  NumPy, doesn't work with PyTorch.  So we're just going to use arrays  instead of tensors for a moment. 
Now have a look at this.  If I try to do a dot product of 1, 2,  3 and 2, 3, 4, it's pretty easy to do. 
It took a fifth of a second,  which sounds terrible.  But the reason it took a fifth of a  second is because that's actually how  
long it took to compile this and run it. Now that it's compiled, the second time,  
it just has to call it. It's now 21 microseconds.  And so that's actually very fast. So with Numba, we can basically make  
Python run at C speed. So now the important thing to  
recognize is if I replace this loop in Python with  a call to dot, which is running in machine code,  
then we now have one, two loops  running in Python, not three. 
So our 448 milliseconds. Well, first of all, let's make sure if I run it,  
run that matmul, it should be close to my t1. t1 is what we got  
before, remember? So when I'm refactoring or  performance improving or whatever, I always like  
to put every step in the notebook and then test. So this test_close() comes from fastcore.test and  
it just checks that two things are very similar. They might not be exactly the same because of   little floating point differences, which is fine. Okay. 
So our matmul is working correctly, or at  least it's doing the same thing it did before.  So if we now run it, it's taking 268  microseconds, okay, versus 448 milliseconds. 
So it's taking about 2,000 times faster  just by changing the one innermost loop. 
So really all we've done is we've added  @njit to make it 2,000 times faster. 
So Numba is well worth knowing about. It can make your Python code very, very fast. 
Okay. Let's keep making it faster. 
So we're going to use stuff again,  which kind of goes back to APL. 
And a lot of people say that learning  APL is a thing that's taught them   more about programming than anything else. So it's probably worth considering learning APL. 
And let's just look at these various things. We've got a is 10, 6, -4. 
So remember at APL, we don't say equals. Equals actually means equals funnily enough.  We, to say set to, we use this arrow  and it's a, this is a list of 10, 6, 4. 
Okay. And then b is 2, 8, 7. 
Okay. And we're going to add them up a+b.  
So what's going on here? So it's  really important that you can think of  
a symbol like a as representing a tensor  or an array. APL calls them arrays, Pytorch  
calls them tensors, NumPy calls them arrays. They're the same thing.  So this is a single thing that  contains a bunch of numbers. 
This is an operation that  applies to arrays or tensors.  And what it does is it works  what's called element wise. 
It takes each pair 10 and 2 and adds them  together, each pair 6 and 8, add them together. 
This is element wise addition. And Fred's asking in the chat, how do you put,   put in these symbols? If you just mouse over  any of them, it will show you how to write it. 
And the one you want is  the one at the very bottom.  The very bottom, which is  the one where it says prefix. 
Now the prefix is the backtick character. So here it's saying prefix hyphen gives us times. 
So if I type P hyphen, there we go. So I've got a   back tick dash b is a x b for example. So yeah, they all have shortcut keys, which you  
learn pretty quickly. I find.  And there's a fairly consistent kind  of system for those shortcut keys too. 
All right. So we can do the same thing in PyTorch.  It's a little bit more verbose in PyTorch,  which is one reason I often like to do  
my mathematical fiddling around in APL. I can often do it with less boilerplate,  
which means I can spend more time thinking, you  know, I can see everything on the screen at once.  I don't have to spend as much time trying  to like ignore the tensor around bracket  
square bracket dot comma blah, blah, blah. It's all cognitive load, which I'd rather ignore.  But anyway, it does the same thing. So I can say a + b and it works exactly like APL. 
So here's an interesting example. I can go (a < b).float().mean(). 
So let's try that one over here. a < b. So this is a really important   idea, which I think was invented by Ken  Iverson, the APL guy, which is the true  
and false represented by zero and one. And because they're represented by  
zero and one, we can do things to them. We can add them up and subtract them and so forth. 
It's a really important idea. So in this case, I want to take the mean of them  
and I'm going to tell you something amazing, which  is that in APL, there is no function called mean. 
Why not? That's because we can  write the mean function, which  
is… so that's four letters, mean, m-e- a-n. We can write the mean function from scratch  
with four characters. I'll show you.  Here's the whole mean function. We're going to create a function  
called mean and the mean is equal to the sum  of a list divided by the count of a list. 
So this here is sum divided by count. And so I have now defined a new function   called mean, which calculates the mean,  mean of a is less than b. There we go. 
And so, you know, in practice, I'm not sure  people would even bother defining a function   called main because it's just as easy to  actually write its implementation in APL.  
In NumPy or whatever Python, it's going to take  a lot more than four letters to implement mean. 
So anyway, it's, you know, it's a math notation. And so being a math notation, we can do a lot   with little, which I find helpful because  I can see everything going on at once. 
Anywho. Okay.  So that's how we do the same thing in PyTorch. And again, you can see that the less than in both  
cases are operating element wise. Okay.  So a is less than b is saying 10 is less than 2, 6  is less than 8, 4 is less than 7 and gives us back  
each of those trues and falses as zeros and ones. And according to the emoji on our YouTube chat,  
Siva's head just exploded as it should. This is why APL is life changing. 
Okay. Let's now go up to higher ranks.  So this here is a rank one tensor. So a rank 1 tensor means it's a list of things. 
It's a vector. It's where else   a rank 2 tensor is like a list of lists. They all have to be the same length lists,  
or it's like a rectangular bunch of numbers  and we call it, in math, we call it a matrix.  So this is how we can create a tensor  containing 1, 2, 3 4, 5, 6 7, 8, 9. 
And you can see often what I like to do  is I want to print out the thing I just   created after I created it. So two ways to do it. 
You can say, put an enter and then  write m and that's going to do that.  Or if you want to put it all on  the same line, that works too. 
You just use a semicolon. Neither one's better than the other.  They're just different. So we could do the same thing in APL. 
Of course, in APL, it's going to be much easier. So we're going to define a matrix called m  
which is going to be a 3 by 3 tensor  containing the numbers from 1 to 9. 
Okay and there we go. That's done it in APL.  A 3 by 3 tensor containing  the numbers from 1 to 9. 
A lot of these ideas from APL you'll find have  made their way into other programming languages.  For example, if you use Go,  you might recognize this. 
This is the Iota character  and Go uses the word Iota. 
They spell it out in a somewhat similar way.  A lot of these ideas from APL have found  themselves into math notation and other languages. 
It's been around since the late 50s. Okay so here's a bit of fun. 
Frobenius norm
We're going to learn about a new thing that  looks kind of crazy called Frobenius norm. 
And we'll use that from time to time  as we're doing generative modeling.  And here's the definition of a Frobenius norm. It's the sum  
over all of the rows and columns of a matrix. And we're going to take each one and square it. 
We're going to add them up and  they're going to take the square root.  And so to implement that in PyTorch is as simple  as going n times m dot sum dot square root. 
So this looks like a pretty complicated  thing when you kind of look at it at first. 
It looks like a lot of squiggly business. Or if you said this thing here, you might be   like, what on earth is that? Well now you  know it's just square, sum, square root. 
So again, we could do the same thing in APL. 
So let's do, so in APL we want the, okay, so  we're going to create something called sf. 
Now it's interesting, APL does  this a little bit differently.  So dot sum by default in  PyTorch sums over everything. 
And if you want to sum over just one dimension,  you have to pass in a dimension keyword. 
For very good reasons APL is the opposite. It just sums across rows or just down columns. 
So actually we have to say sum up the flattened   out version of the matrix and to  say flattened out you use comma. 
So here's sum up the flattened  out version of the matrix. 
Okay so that's our sf, oh sorry, and  the matrix is meant to be m times m. 
There we go. So there's the same thing.  Sum up the flattened out m by m matrix. 
And another interesting thing about  APL is it always is read right to left.  There's no such thing as operator  precedence, which makes life a lot easier. 
Okay and then we take the square root of that.  There isn't a square root function,  so we have to do to the power of 0.5. 
And there we go, same thing. All right, you get the idea.  Yes, so very interesting  question here from mariboo. 
Are the bars for norm or absolute value,  and I like Siva's answer, which is the norm  
is the same as the absolute value for a scalar. So in this case you can think of it as absolute   value and it's kind of not needed  because it's being squared anyway. 
But yes, in this case the norm, well in every case  for a scalar the norm is the absolute value, which  
is kind of acute discovery when you realize it. So thank you for pointing that out Siva. 
All right, so this is just  fiddling around a little bit to   kind of get a sense of how these things work. So really importantly, you can index into a  
matrix and you'll say rows first and then columns. And if you say colon, it means all the columns. 
So if I say row 2, here it  is row 2, all the columns,  
sorry, this is row 2, so that's 0 —APL starts  at 1 or the columns that's going to be 7, 8, 9. 
And you can see, I often use comma to print  out multiple things and I don't have to say   print in, in Jupiter. It's kind of assumed. 
And so this is just a quick way of printing out  the second row and then here, every row column 2. 
So here is every row of column 2. And here you can see 3, 6, 9. 
So one thing very useful to recognize is that for  tensors of higher rank than 1, such as a matrix,  
any trailing colons are optional. So you see this here, m[2],  
that's the same as m[2, :]. It's really important to remember.  So m[2], you can see the result is the same. So that means row 2, every column. 
So now with all that in place,   we've got quite an easy way. We don't need a number anymore. 
We can multiply. So we can get  rid of that innermost loop.  So we're going to get rid of this loop because  this is just multiplying together all of the  
corresponding rows of a with the, sorry, all the  corresponding colons of a row of a with all the  
corresponding rows of a column of b. And so we  can just use an element wise operation for that. 
So here is the ith row of a, and here is the jth  column of b And so those are both, as we've seen,  
just vectors, and therefore we can do an element  wise multiplication of them and then sum them up. 
And that's the same as a dot product. So that's handy. 
And so again, we'll do test_close. Okay.  It's the same. Great.  And again, you'll see, we kind of  did all of our experimenting first,  
right? To make sure we understood how  it all worked and then put it together.  And then if we time it: 661 microseconds. Okay. 
So it's interesting. It's actually slower than, which really   shows you how good Numba is, but it's certainly  a hell of a lot better than our 450 milliseconds. 
But we're using something that's  kind of a lot more general now. 
This is exactly the same  as dot as we've discussed.  So we could just use torch dot,  torch.dot(), I suppose I should say. 
And if we run that, okay. Little faster.  It's still, interestingly, it's still slower  than the Numba, which is quite amazing actually. 
All right. So that's… that   one was not exactly a speed up, but it's  kind of a bit more general, which is nice. 
Now we're going to get something into  something really fun, which is broadcasting. 
Broadcasting with scalars and matrices
And broadcasting is about what if you have arrays  with different shapes? So what's a shape? The  
shape is the number of rows or the number of rows  and columns or the number of, what would you say,  
faces, rows and columns and so forth. So for example, the shape of m is 3 by 3. 
So what happens if you multiply or add or do  operations to tensors of different shapes?  
Well, there's one very simple one, which is  if you've got a rank 1 tensor, the vector,  
then you can use any operation with a scalar  and it broadcasts that scalar across the tensor. 
So a > 0 is exactly the same as saying  a is greater than tensor([0, 0, 0])  
So it's basically copying that across three times. Now it's not literally making a copy in memory,  
but it's acting as if we had said that. And this is the most simple version of   broadcasting. Okay. 
It's broadcasting the 0 across the 10 and the 6  and the -4 and APL does exactly the same thing. 
A < 5, so 0, 0, 1. So it's the same idea. 
Okay. So we can do plus with a scalar and we can do  
exactly the same thing with higher than rank one. So two times a matrix is just going to do:   two is going to be broadcast across  all the rows and all the columns. 
Okay. Now it gets interesting.  So broadcasting dates back to APL, but a really  interesting idea is that we can broadcast not  
just scalars, but we can broadcast vectors across  matrices or broadcast any kind of lower ranked  
tensor across higher rank tensors, or even  broadcast together two tensors of the same rank,  
but different shapes and a really powerful way. And as I was exploring this, I was trying to,  
I love doing this kind of computer archeology. I was trying to find out where   the hell this comes from. But it actually turns out from this email  
message in 1995 that the idea actually comes from  a language that I'd never heard of called Yorick,  
which still apparently exists. Here's Yorick. 
And so Yorick has talked about  broadcasting and conformability.  So what happened is this very obscure  language has this very powerful idea  
and NumPy has happily stolen the idea from  Yorick that allows us to broadcast together  
tensors that don't appear to match. So let me give an example.  Here's a tensor called c that's a vector. It's a rank 1 tensor, 10, 20, 30, and here's  
a tensor called m, which is a matrix. We've seen this one before,   and one of them is shape 3 comma 3. The other is shape 3. 
And yet we can add them together. 
Now, what's happened when we added  it together? Well what's happened  
is 10, 20, 30 got added to 1, 2, 3, and then  10, 20, 30 got added to 4, 5, 6, and then 10,   20, 30 got added to 7, 8, 9. And hopefully you can  
see this looks quite familiar. Instead of broadcasting a scalar over a  
higher rank tensor, this is broadcasting  a vector across every row of a matrix. 
And it works both ways, so we can say  c + m gives us exactly the same thing. 
And so let me explain what's  actually happening here.  The trick is to know about this somewhat  obscure method called expand_as(). 
And what expand_as() does is this  creates a new thing called t,   which contains exactly the same thing  as c, but expanded or kind of copied  
over so it has the same shape as  m. So here's what t looks like.  Now t contains exactly the same thing as c  does, but it's got three copies of it now. 
And you can see we can definitely  add t to m because they match shapes. 
So we can say m plus t, we know we can play  m plus t because we've already learned that  
you can do element wise operations on  two things that have matching shapes. 
Now by the way, this thing t didn't  actually create three copies.  Check this out. If we call t dot storage,  
it tells us what's actually in memory. It actually just contains the numbers 10, 20, 30,  
but it does a really clever trick. It has a stride of 0 across the  
rows and a size of 3 comma 3. And so what that means is that   it acts as if it's a 3 by 3 matrix. And each time it goes to the next row,  
it actually stays exactly where it is. And this idea of strides is the trick  
which NumPy and PyTorch and so forth use for all  kinds of things where you basically can create  
very efficient ways to do things like expanding or  to kind of jump over things and stuff like that,  
switch between columns and rows, stuff like that. Anyway, the important thing here for us to   recognize is that we didn't actually make a copy. We made it completely efficient and it's all  
going to be run in C code very fast.  So remember this expand_as is critical. This is the thing that will teach you to  
understand how broadcasting works, which  is really important for implementing deep  
learning algorithms or any kind of  linear algebra on any Python system  
because the NumPy rules are used exactly the same  in JAX, in TensorFlow, in PyTorch and so forth. 
Now I'll show you a little trick, which  is going to be very important in a moment.  If we take c, which remember is  a vector containing 10, 20, 30,  
and we say .unsqueeze(0), then it  changes the shape from 3 to 1 comma 3. 
So it changes it from a vector of length  3 to a matrix of 1 row by 3 columns. 
This will turn out to be  very important in a moment.  And you can see how it's printed. It's printed out with two square brackets. 
Now I never use unsqueeze because I much  prefer doing something more flexible,   which is if you index into an access with a  special value, None, also known as np.newaxis(),  
it does exactly the same thing. It inserts a new axis here.  So here we'll get exactly the same thing,  1 row by all the columns, 3 columns. 
So this is exactly the same as saying unsqueeze.  So this inserts a new unit axis. This is a unit axis, a single row  
in this dimension, and this does the same thing. So these are the same. 
So we could do the same thing and say,  unsqueeze(1), which means now we're  
going to unsqueeze into the first dimension. So that means we now have 3 rows and 1 column. 
See the shape here, the shape is inserting a  unit axis in position 1, 3 rows and 1 column. 
And so we can do exactly the same thing here. Give us every row and a new unit axis  
in position 1, same thing. So those two are exactly the same. 
So this is how we create a matrix with 1 row. This is how we create a matrix with 1 column. 
None comma colon versus colon  comma None or unsqueeze. 
We don't have to say, as we've learned before,  None comma colon, because, do you remember?  
Trailing colons are optional.  So therefore just c[None] is also going  to give you a row matrix, 1 row matrix. 
This is a little trick here. If you say dot, dot, dot,   that means all of the dimensions. And so dot, dot, dot comma None will  
always insert a unit axis at the end,  regardless of what rank a tensor is. 
So yeah, so None and np new  mean exactly the same thing.  np newaxis is actually a synonym for None. If you've ever used that, I always use None  
because why not? Short and simple. So here's something interesting.  If we go c[:, None], so let's go and  check out what c[:, None] looks like. 
c[:, None] is a column. 
And if we say expand as m, which is 3  by 3, then it's going to take that 10,   20, 30 column and replicate it. 10, 20, 30; 10, 20, 30; 10, 20, 30. 
So we could add, so remember like, well, remember  I will explain that when you say matrix plus  
c[:, None], it's basically going  to do this .expand_as() for you. 
So if I want to add this matrix here to m,   I don't need to say .expand_as() I just  write this, I just write m + c[:, None]. 
And so this is exactly the same as doing m + c,  but now rather than adding the vector to each row,  
it's adding the vector to each column. So you plus 10, 20, 30; 10, 20, 30; 10, 20, 30. 
So that's a really simple way that we now get kind  of a free, thanks to this really nifty notation. 
There's a nifty approach that came from Yorick. So here you can see m + c[None, :] is adding 10,  
20, 30 to each row and m + c[:, None]  is adding 10, 20, 30 to each column. 
All right, so that's the  basic like hand wavy version.  So let's look at, like, what are  the rules and how does it work?  
Okay, so c[None, :] is 1 by 3. c[:, None] is 3 by 1. 
What happens if we multiply c[None, :] by c[:,  None]? Well it's gonna do, if you think about it,  
which you definitely should,  cause thinking's very helpful. 
What is going on here? Oh, took forever. 
Okay, so what happens if we go c[None, :] times  c[:, None]? So what it's gonna have to do is  
it's gonna have to take this 10, 20, 30 column  vector or 3 by 1 matrix, and it's gonna have to  
make it work across each of these rows. So what it does is expands it to be 10,  
20, 30; 10, 20, 30; 10, 20, 30. So it's gonna do it just like this. 
And then it's gonna do the  same thing for c[None, :].  So that's gonna become three rows of 10, 20, 30. So we're gonna end up with 3 rows of 10, 20, 30  
times 3 columns of 10, 20,  30, which gives us our answer. 
And so this is gonna do an outer product. So it's very nifty that you can actually  
do an outer product without any special  functions or anything just using broadcasting. 
And it's not just outer products. You can do outer Boolean operations.  And this kind of stuff comes up  all the time, right? Now remember,  
you don't need the comma colon, so get rid of it. So this is showing us all the places where  
it's greater than, it's kind of an outer  Boolean, if you wanna call it that. 
So this is super nifty and you can do all kinds of  tricks with this because it runs very, very fast. 
So this is gonna be accelerated in C. So here are the rules. 
Broadcasting rules
When you operate on two arrays of tensors,  NumPy and PyTorch will compare their shapes. 
So remember the shape, this is a shape. You can tell it's a shape because we said shape  
and it's goes from right to left. So that's the trailing dimensions. 
And it checks whether dimensions are compatible. Now they're compatible if they're equal,   right? So for example, if we say m * m,  then those two shapes are compatible because  
in each case, it's just gonna be 3,  right? So they're gonna be equal.  So if the shape in that dimension  is equal, they're compatible. 
Or if one of them is 1. And if one of them is 1,   then that dimension is broadcast to  make it the same size as the other. 
So that's why the outer product worked. We had a 1 by three times a 3 by 1. 
And so this 1 got copied 3  times to make it this long.  And this 1 got copied 3  times to make it this long. 
Okay, so those are the rules. So the arrays don't have to   have the same number of dimensions. So this is an example that comes up all the time. 
Let's say you've got a 256 by 256  by 3 array or tensor of RGB values.  So you've got an image in  other words, a color image. 
And you want to normalize it. So you want to scale each color   in the image by a different value. So this is how we normalize colors. 
So one way is you could multiply  or divide or whatever, multiply   the image by a 1 dimensional array with 3 values. 
So you've got a 1D array. So that's just 3.  Okay. And then the image is 256 by 256 by 3. 
And we go right to left, and we check  are they the same? We say yes, they are. 
And then we keep going left and we say  are they the same? And if it's missing,   we act as if it's one. And if we keep going,  
if it's missing, we act as if it's one. This is going to be the same as doing 1 by 1 by 3. 
And so this is going to be broadcast,  these 3 elements will be broadcast  
over all 256 by 256 pixels. So this is a super fast and  
convenient and nice way of normalizing  image data with a single expression.  And this is exactly how we do it  in the fastai library, in fact. 
Matrix multiplication with broadcasting
So we can use this to dramatically  speed up our matrix multiplication. 
Let's just grab a single  digit, just for simplicity.  And I really like doing this in Jupyter Notebooks. And if you build Jupyter Notebooks to explain  
stuff that you've learned in this course or ways  that you can apply it, consider doing this for   your readers, but add a lot more prose. I haven't added prose here because I  
want to use my voice. If I was, for example,   in our book that we published, it's all written in  notebooks and there's a lot more prose, obviously. 
But like really, I like to show every example  all along the way using simple as possible. 
So let's just grab a single digit. So here's the first digit.  So its shape is, it's a 784 long vector. Okay. 
And remember that our weight matrix is 784 by 10. 
Okay. So if we say digit[:, None] dot shape,  
then that is a 784 by 1 row matrix. Okay. 
So there's our matrix. And so if we then  
take that 784 by 1 and expand as m2, it's going  to be the same shape as our weight matrix. 
So it's copied our image data for  that digit across all of the 10  
vectors representing the 10 kind of linear  projections we're doing for our linear model. 
And so that means that we  can take the digit[:, None].  So 784 by 1 and multiply it by the weights. And so that's going to get us back 784 by 10. 
And so what it's doing, remember,  is it's basically looping through   each of these 10 784 long vectors. 
And for each one of them, it's  multiplying it by this digit. 
So that's exactly what we want to  do in our matrix multiplication. 
So originally we had, well, not originally,  most recently, I should say, we had this  
dot product where we were actually looping over  j, which was the columns of b. So we don't have  
to do that anymore because we can do it all  at once by doing exactly what we just did.  So we can take the ith row and all the  columns and add an axis to the end. 
And then just like we did here,  multiply it by b. And then dot sum. 
And so that is, again, exactly the same thing. That is another matrix multiplication,  
doing it using broadcasting. Now this is, like, tricky to get your head around. 
And so if you haven't done this  kind of broadcasting before,   it's a really good time to pause the  video and look carefully at each of these  
four cells before and understand what did I do  there? Why did I do it? What am I showing you?  
And then experiment with trying to, and  to remember that we started with m1[0],  
right? So just like we have here a[i]I. So that's why we've got [i, :, None] , because  
this digit is actually m1[0]. This is like m1, zero, colon, none. 
So this line is doing exactly the  same thing as this here, plus a sum. 
So let's check if this matmul is the same  as it used to be, yet it's still working.  And the speed of it, okay, not bad. So 137 microseconds. 
So we've now gone from a time from 500  milliseconds to about 0.1 milliseconds. 
Funnily enough on my, oh, actually now I  think about it, my MacBook Air is an M2,   whereas this Mac Mini is an M1. So that's a little bit slower. 
So my Air was a bit faster than 0.1 milliseconds.  So overall, we've got about a  5,000 times speed improvement. 
So that is pretty exciting. And since it's so fast now,   there's no need to use a mini batch anymore. If you remember, we used a mini batch of,  
where is it, of five images. But now we can actually use the  
whole data set because it's so fast. So now we can do the whole data set. 
There it is. We've now got 50,000 by 10,   which is what we want. And so it's taking us only 656  
milliseconds now to do the whole data set. So this is actually getting to a point now   where we could start to create and train some  simple models in a reasonable amount of time. 
So that's good news. All right. 
I think that's probably a  good time to take a break.  We don't have too much more of this to go,  but I don't want to keep you guys up too late. 
So hopefully you learned something  interesting about broadcasting today.  I cannot overemphasize how widely useful this is  in all deep learning and machine learning code. 
It comes up all the time. It's basically our   number one most critical kind  of foundational operation. 
So yeah, take your time practicing it  and also good luck with your diffusion  
homework from the first half of the lesson. Thanks for joining us and I'll see you next time.

---

# 12

Hi everybody, welcome back to Lesson 12  of Practical Deep Learning for Coders. 
So got a lot of stuff to cover  today, so let's dive straight in.  And I actually thought I would  start by sharing something which  
CLIP Interrogator & how it works
I've seen been getting a lot of attention  recently, which is the CLIP Interrogator. 
So the CLIP Interrogator is a Hugging Face Spaces,  I guess, Gradio app where I uploaded my image here  
and its output, let's just zoom in a bit,  
its output a text prompt for creating  a CLIP embedding from, I guess. 
So I've seen a lot of folks on  Twitter and elsewhere on the internet  
saying that this is producing the CLIP  prompt that would generate this image. 
And generally speaking, the CLIP, the  prompts it creates are rather rude. 
My one's less rude than some, although, you  know, extremely long forehead, maybe not,   thanks very much, but your personal  data avatar, funny professional photo. 
I don't know what tectonics has  been to me here without eyebrows.  So this doesn't actually return the CLIP  prompt that would generate this photo at all. 
And the fact that some people are saying that  makes me realize that some people have no idea  
what's going on with Stable Diffusion. So I thought we might take this as an   opportunity to explain why we can't do  that and what we can try and do instead. 
So let's imagine that my friend took a photo and  —of himself— and he wanted to send me his photo  
and he thought he would compress it a whole lot. So what he did was he put it through the  
CLIP image encoder.  Okay. So that's going to take this big image  
and it's going to turn it into an embedding. And the embedding is much,  
much smaller than the image. It's just a vector of a few floats. 
So then my friend hopes that they  could send me this embedding. 
And so they send that over in an email and they  say, there you go, Jeremy, there's the CLIP   embedding of the photo I wanted to send you. So now you just have to  
decode it to turn it back into a picture. 
So now I've got the embedding  and I have to decode it.  How would you do that?  
Well, you can't. We have a function here, let's call it f, which  
is the clip image encoder, which takes as input an  image, which I'll call x and returns an embedding. 
Does that mean that there is some other function  and inverse functions —we normally write with a  
minus one—, an inverse function with which I can  take that embedding, let's say we call that y,  
we pass it y and it would give us back our photo. 
And so y, remember, is f(x). So to put it another way,   this is f inverse of f of f of y. So an inverse function is something that  
undoes a function and so that gives you back y. Is there an inverse function  
for the CLIP image encoder? Well, not  everything has an inverse function. 
For example, consider the function  like, let's say in Python,  
which takes def f of x, returns zero. Can you invert that function,  
if you get back, you pass in three, you get  back zero, is there a function, oopsie, zero,  
is there a function that's gonna take the output  and give you back the input? No, of course not,   because you just threw the whole thing away. So not all functions can be inverted. 
And indeed, in this case, we've started with a  function, which is whatever 512 by 512 by 3, say,  
and we've turned it into  something much, much smaller.  I can't remember exactly how  big a CLIP image encoding is,  
embedding is, but it's much smaller. So clearly we're losing something. 
But what I could do is I could put  it through a diffusion process. 
And so remember, a diffusion process is  something where we have learned, we have  
taught or… we shouldn't, I don't know if taught,  well an algorithm has learned to take some noise. 
So we could start with some noise, and  we could start with an image embedding.  We haven't done this before, but we could do that. We could train something that takes noise and  
image embedding and removes a bit of the noise. And we could run that a bunch of times. 
And it wouldn't give us back the original picture,  but hopefully it would give us something back  
if it's a conditional. So remember, using the conditional   diffusion approach, we'd get back something  that might be something like our original image. 
So that's what diffusion is, right? Diffusion  is something that takes an embedding  
and inverts an encoder to give you back something  that hopefully might generate that embedding. 
Now, of course, remember, we don't  actually get image embeddings when we do   prompts in stable diffusion. Instead, we have text embeddings. 
But if you remember, that actually doesn't matter. Because do you remember how we actually,  
well “we”, OpenAI, trained CLIP so that they  had various pictures along with their captions,  
and they trained an algorithm that was explicitly  designed to make it so that each image returned a  
embedding for the image that  was similar to the embedding   that the text encoder created for the caption. 
And remember, all of the stuff that didn't  match, it was trained to be different.  And so that means that a text  embedding, which describes this picture,  
and the actual image embedding of this picture  should be very similar if they're CLIP embeddings. 
That's the definition of CLIP embeddings. So you see this idea that you could take  
a text or image embedding and turn it back  into an image perfectly makes no sense. 
This is the very definition of the thing  we're trying to do when we do CLIP.  And because what we're basically trying  to do is invert the embedding function,  
these kinds of problems are generally  referred to as “inverse problems”. 
So Stable Diffusion is something that attempts to  approximate the solution to an inverse problem. 
So why does that mean that CLIP interrogator is  not actually inverting the picture to give us back  
the text? Well, it's just as nonsensical. If we've got an image embedding,  
right, trying to undo that to get back to the  picture and trying to undo that to get back  
to a suitable prompt is equally infeasible. Both of them require inverting an encoder. 
And that just doesn't exist. The best we can do is, or at   least the best we know how to do at the moment,  is to approximate that using a diffusion process. 
Okay, so that's why these texts that it spits  back are fun and interesting, but they are not  
the thing that you can put back into Stable  Diffusion and have it generate the same photo. 
And the nice thing is that actually  the code for this is available. 
And you can take a look at it. Here's the app. 
And you'll see what it does is it has a big  list of —let's have a look at some examples. 
So it has a big, this has big lists of  examples, for example, a big list of artists. 
And it has a big list of mediums and  a big list of movements, and so forth. 
It's got all this hard coded pieces of text. And so what it does is it basically mixes  
and matches those various things  together to see which one works well. 
And it combines it with the output of  something called the BLIP language model,  
which is not designed to give you an  exactly accurate description of an image,  
but it has been specifically trained to give an  okay-ish caption for an image and it actually  
works reasonably well. But again, it's not   it's not the inverse of the clip encoder. So okay, so that's how that all works. 
Matrix multiplication refresher
So where we had got to was that we had done  matrix multiplication with broadcasting,  
where we had broadcast the entire column  from the right hand matrix all at once. 
And that allowed us to get it down to a point  where we only have one for loop written in Python. 
And generally speaking, we do not want to be  doing loop— looping through too many things   in Python, because that's the slow bit. So the two inner loops we originally had,  
which just to remind us,  
originally were here, these two inner loops,  looping through 10, and then to 784 respectively,  
have been replaced with a single line of code.  So that was pretty great. And our times now is  
increased— is improved by 5,000 times. So we're 5,000 times faster than we started out. 
So another trick that we can use, which I'm a big  fan of, is something called Einstein summation. 
Einstein summation
And Einstein summation is a compact  representation for representing products and sums. 
And this is an example of an Einstein summation. And what we're going to do now is we're going to  
replicate our matrix product  with an Einstein summation.  And believe it or not, the entire  thing can be pushed down to just  
these characters, which is pretty amazing. So let me explain what's happening here.  The arrow is separating the left  hand side from the right hand side. 
The left hand side is the inputs. The right hand side is the output.  The comma is between each input. So there are two inputs. 
The letters are just names that you're giving  to the number of rows and the number of columns. 
So the first matrix we're multiplying  by has i rows and k columns. 
The second has k rows and j columns. It's going to go through a process which creates a  
new matrix that —actually this is not even doing,  this is not yet doing the matrix multiplication. 
This is without the sum. This one's going to create a   new matrix that contains i rows and k, well, how  do we say it? i faces and k rows and j columns. 
So a rank three tensor. So the number of letters is going to be the rank. 
And the rules of how this works is that if  you repeat letters between input arrays,  
so here's my inputs, ik and kj,  we've got a repeated letter. 
It means that values along those  axes will be multiplied together.  So it means that each item in each row of,  sorry, in each, yeah, across a row will be  
multiplied by each item down each column  to create this i by k by j output tensor. 
So to remind you, our first matrix is 5 by 784. That's m1. 
Our second matrix is 7084 by 10. That's m2. 
So i is 5, k is 784 and j is 10. So if I do this torch.einsum,  
then I will end up with a i by k by j. It'll be 5 by 784 by 10. 
And if you have a look, I've run it  here on these two tensors, m1 and m2,   and the shape of the result is 5 by 784 by 10. And what it contains is the original 5 rows of m1,  
the original 10 columns of m2, and  then for the other 784, that dimension,  
they're all multiplied together because it's been  copied between the two arguments to the einsum. 
And so if we now sum up that  over this dimension, we get back. 
So what we get back, if we go back to  the original matrix multiply we do,  
we had 10.94, negative, negative 0.68, et cetera. And so now with this Einstein summation version,  
we've got back exactly the same thing. Because what it's done is it's taken each of these  
columns by rows, multiplied them together to  get this 5 by 784 by 10, and then added up  
that 784 for each one, which is exactly  what matrix multiplication does. 
But we're going tu use one of the  two things from Einstein summation.  The second one says if we omit a letter from the  output, so the bit on the right of the arrow,  
it means those values will be summed. So if we remove this k, which gives us ik and  
kj goes to ij, so we've removed the k entirely,  that means that sum happens automatically. 
So if we run this, as you see, we get  back again, matrix multiplication. 
So Einstein summation notation is,  you know, it takes some practice  
getting used to, but it's very convenient. And once you get used to it, it's actually a  
really nice way of thinking about what's going on. And as we'll see in lots of examples,  
often you can really simplify your code by  using just a tiny little Einstein summation. 
And it doesn't even have to be a sum, right?  You can, you don't have to omit any letters   if you're just doing products. So maybe it's a bit misnamed. 
So we can now define our matmul  as simply this torch.einsum. 
So if we now check it, test_close  that the original result is equal  
to this new matmul. And yes, it is.  And let's see how the speed looks. 15 milliseconds. 
Okay. And that was for the whole thing. 
So compared to 600 milliseconds. So as you can see, this is much faster than   even the very fast broadcasting approach we used. So this is a pretty good trick, is torch.einsum. 
Okay. But of course,   we don't have to do any of those things  because PyTorch already knows how to do matmul.  So there's two ways we can run  matmul directly in PyTorch. 
You can use a special at (@) operator. So x_train @ weights is the same as  
matmul(x_train, weights), as you see, test_close. Or you can say torch.matmul. 
And interestingly, as you can see here,  the speed is about the same as the einsum. 
So there's no particular harm, no  particular reason not to do an einsum.  So when I say einsum, that stands  for Einstein summation notation. 
All right. Let's go faster still.  Currently we're just using my CPU. 
Matrix multiplication put on to the GPU
But I have a GPU. It would be nice to use it.  So how does a GPU work? An Nvidia GPU, and indeed,  pretty much all GPUs, the way they work is that  
they do lots and lots of things in parallel. And you have to actually tell the GPU  
what are all the things you want  to do in parallel, one at a time.  And so what we're going to do is we're going  to write in pure Python something that works  
like a GPU, except it won't actually be  in parallel, so it won't be fast at all.  But the first thing we have to do if we're going  to get something working in parallel is we have  
to create a function that can calculate  just one thing, even if a thousand other  
things are happening at the same time,  it won't interact with anything else.  And there's actually a very easy way to think  about matrix multiplication in this way,  
which is what if we try to create something which  just as we've done here fills in a single item  
of the result? So how do we create something  that just fills in row zero, column zero?  
Well what we could do is we could create a  new matmul where we're going to pass in the   coordinates of the place that we want to fill in. So we're going to start by passing it (0, 0). 
We'll pass it the matrix— matrices we want to  multiply, and we'll pass in a tensor that we've  
pre-filled in with zeros to put the result into. So we're going to say, okay, the result   is torch.zeros, rows by columns,  call matmul for location (0, 0),  
passing in those two matrices and the bunch  of zeros matrix ready to put the result in. 
And if we call that, we get  the answer in cell (0, 0).  So here's an implementation of that. So the implementation is first of all,  
we've been passed the (0, 0) coordinates. So let's destructure them.  So hopefully you've been experimenting with  destructuring because it's so important. 
You see it all the time into i and j. That's the row and the column. 
Make sure that that is inside  the bounds of our output matrix. 
And we're going to start by start at zero  and loop through all of the rows of a  
and all of the columns of b for i and j. Sorry, all of the columns of a and all of  
the rows of b for i and j, just like the very  innermost loop of our very first Python attempt. 
And then at the end, pop that into the output. So here's something that fills in   one piece of the grid successfully. So we could call this rows by columns times,  
each time passing in a different grid. And we could do that in parallel because   none of those different locations  interact with any other location. 
So something which can calculate  a little piece of an output on  
a GPU is called a kernel. So we'd call this a kernel.  And so now we can create  something called launch_kernel. 
We pass it the kernel. So that's the function.  So here's an example, launch_kernel  passing in the function. 
And how many rows and how many  columns are there in the output grid. 
And then give me any arguments  that you need to calculate it.  So in Python, *args just says any  additional arguments that you pass  
are going to be put into an array called args. If you use something like C, you might've seen  
like variadic arguments or parameters. It's the same basic idea. 
So we're going to be calling launch_kernel. We're going to be saying launch the kernel matmul,  
using all the rows of a, all the  columns of b, and then the args,  
which are going to be the *args, are going to  be m1, the first matrix, m2, the second matrix,   and res, another torch.zeros we just created. So launch_kernel is going to loop through  
the rows of a, and then for each row of  a, it'll loop through the columns of b  
and call the kernel, which is matmul, on that  grid location passing in m1, m2, and res. 
So *args here is going to unpack that and  pass them as three separate arguments. 
And if I run that, run all of that,   you'll see it's done it. It's filled in the exact same matrix. 
Okay, so that's actually not fast at all. It's not doing anything in parallel,   but it's the basic idea. So now to actually do it in  
parallel, we have to use something called CUDA. So CUDA is a programming model for Nvidia GPUs. 
And to program in CUDA from Python, the  easiest way currently to do that is with  
something called Numba. And Numba is a compiler.  Oh, well, you've seen it  actually already for non-GPU. 
It's a compiler that takes Python code and  spits out, you know, compiled fast machine code. 
If you use its CUDA module, it'll actually  spit out GPU accelerated CUDA code. 
So rather than using at @njit  like before, we now say @cuda.jit  
And it behaves a little bit differently,  but you'll see that this matmul,   let me copy the other one over so you can  compare, compare it to our Python one. 
Our Python matmul and this @cuda.jit matmul  look, I think identical, except for one thing,  
instead of passing in the grid, there's a special  magic thing called cuda.grid And you say, how many  
dimensions does my grid have? And you unpack it. So that's, you don't have to, it's just a little   convenience that Numba does for you. You don't have to pass over the grid. 
It passes it over for you. So it doesn't need this grid.  Other than that, these two are identical, but the  decorator is going to compile that into GPU code. 
So now we need to create our  output tensor just like before,  
and we need to do something else, which is we  have to take our input matrices and our output. 
So our input tensors, the matrices  in this case, and the output tensor,   and we have to move them to the GPU. Or I should say, copy them to the GPU. 
So cuda.to_device copies a tensor to the GPU. And so we've got three things  
getting copied to the GPU here. And therefore we store the three things over here. 
Another way I could have written this  is I could have said map, which I kind  
of quite like doing, a function, which is  cuda.to_device to each of these arguments. 
And this would be the same thing. This is going to call cuda.to_device   on x_train and put it in here on weights and  put it in here and on r and put it in here. 
That's a slightly more convenient way to do it. 
Okay. So we've got our 50,000 by 10 output.  That's just all zeros, of course. That's just how we created it. 
And now we're going to try and fill it in. There is a particular detail that you don't have  
to worry about too much, which is in CUDA. They don't just have a grid,  
but there's also a concept of blocks.  And there's something we call here  TPB, which is threads per block. 
This is just a detail of  the CUDA programming model.  You don't have to worry about too much. You can just basically copy this. 
And what it's going to do is it's going to call  each grid item in parallel with a number of  
different processes, basically. So this is just the code   which turns the grid into blocks. And so you don't have to worry too  
much about the details of that. You just always run it. 
Okay. And so now how do you call   the equivalent of launch_kernel? Well, it's a  slightly weird way to do it, but it works fine. 
You call matmul, but because matmul has  @cuda.jit, it's got a special thing,   which is you have to put something in  square brackets afterwards, which is  
you have to tell it how many blocks per grid. That's just the result from the previous cell   and how many threads per block  in each of the two dimensions. 
So again, you can just copy and paste this  from my version, but then you pass in the   three arguments to the function. This will be a, b,  
and c. And this is how you launch a kernel. So this will launch the kernel matmul on the GPU. 
At the end of it, rg is going to get filled in. It's on the GPU, which is not much good to us. 
So we now have to copy it back to the CPU, which  is called the host, copy to host, to run that. 
And it's done. And test_close shows us that our   result is similar to our original result. So it seems to be working. 
So that's great. I see Siva on the   YouTube chat is finding that  it's not working on his Mac. 
That's right. So this will only work on an NVIDIA GPU,   as basically all of the GPU, nearly all the  GPU stuff we look at only works on NVIDIA GPUs. 
Mac GPUs are gradually starting to get a little  bit of support from machine learning libraries,  
but it's taking quite a while. It's got quite a way to go,   as I say this at least towards the end of 2022. If this works for you later on, that's great. 
Okay. So let's time how fast that is.  Okay. So that was 3.61 milliseconds. 
And so if we compare that to the PyTorch  matmul on CPU, that was 15 milliseconds. 
So that's great. So it's faster still.  So how much faster... Oh, by the way,  
we can actually go faster than that, which  is we can use the exact same code we had  
from the PyTorch up. But here's a trick.  If you just take your tensor and write .cuda()  after it, it copies it over to the GPU. 
If it's on a NVIDIA GPU, do the same for  weights.cuda() So these are our two CUDA versions. 
And now I can do the whole thing. And this will actually run on the GPU. 
And then to copy it back to the host, you just  say .cpu() So if we look to see how fast that is,  
458 microseconds. So that is... 
Somebody just pointed out that  I wrote the wrong thing here.  1e-3. Okay. 
So how much faster is that?  458 microseconds, our original,  
on the whole dataset, was 663 micromiliseconds. So compared to our broadcast version,  
we are another thousand times faster. So overall, this version here  
compared to our original version, which was  
here... Here, the difference in performance is  
5 million X. So when you  
see people say, yeah, Python can be pretty  slow, it can be better to run stuff on the GPU. 
If possible, we're not talking about a 20% change. We're talking about a 5 million X change. 
So that's a big deal.  And so that's why you need to  be running stuff on the GPU. 
All right. Some folks on YouTube are wondering how   on earth I'm running CUDA when I'm on a Mac. And given it says local host here. 
That's because I'm using something called SSH  tunneling, which we might get to sometime. 
I suspect my live coding from the previous  course might have covered that already.  But this is basically you can use a Jupyter  Notebook that's running anywhere in the world  
from your own machine using something called  SSH tunneling, which is a good thing to look up. 
Okay. One person asks if Einstein   summation burrows anything from APL. Yes, it does actually. 
So it's kind of the other way around, actually. APL burrows it from Einstein notation. 
So I don't know if you remember I mentioned  that Ken Iverson, when he developed APL,  
was heavily influenced by tensor analysis. And so this Einstein notation is  
very heavily used there. If you'll notice, a key thing that   happens in Einstein notation is there's no loop. There isn't this kind of sigma, you know,  
i from here to here, and then you put the i  inside the function that you're summing up. 
Everything's implicit. And APL takes that a very long way. 
And J takes it even further, which is  what Ken Iverson developed after APL.  And this kind of general idea of removing  the index is very important in APL,  
and it's become very important in NumPy,  PyTorch, TensorFlow, and so forth. 
So finally, we know how to multiply matrices. Congratulations. 
So let's practice that. Let's practice what we've learned. 
So we're going to go to  02_meanshift to practice this. 
Clustering (Meanshift)
And so we're going to try to exercise our  
kind of tensor manipulation  operation muscles in this section.  And the key actually endpoint  for this is the homework. 
And so what you need to be doing  is getting yourself to a point   that you can implement something like  this, but for a different algorithm. 
Why do we care about this? Because  this is like learning your times table,  
your times tables, if you're doing, you  know, mathematics, it's this kind of like  
thing that's going to come up all the time. And if you're not good at your times tables,  
everything else, a lot, a lot of other things,  particularly at primary school and high school,   you know, they get difficult. You get slower and it's frustrating. 
And you spend time thinking about these mechanical  operations rather than getting your work done. 
It is, it's important that when you have  an idea about something you want to try or   debug or profile or whatever, that you can quickly  translate that into working code and the way that  
code is written for GPUs or even for fast running  on CPUs is using broadcasting, Einstein notation,  
matrix modifications, and so forth. So you've got to, you've got to,  
got to, got to practice super important. So we're going to practice it by  
running, by developing a clustering algorithm  and the clustering algorithm we're going to work  
on is something called mean shift clustering,  which hopefully you've never heard of before. 
And I say that because I just  think it's a really fun algorithm,   but not many people have come across, excuse me. And I think you'll find it really useful. 
So what is cluster analysis? Cluster analysis  is very different to anything that we've worked  
on in this course so far and that there isn't a  dependent variable that we're trying to match. 
But instead we're just trying to find,  are there groups of similar things in this  
data and those groups we call clusters. And as you can see from the Wiki page,   there's all kinds of applications of cluster  analysis across many different areas. 
I will say that sometimes cluster  analysis can be overused or misused. 
It's really best for when your,  
your various columns are the same kind  of thing and have the same kind of scale. 
For example, pixels are  all the same kind of thing.  They're all pixels. So one of the examples  
they use is market research. So I wouldn't use cluster analysis   for socio-demographic inputs because  they're all different kinds of things. 
But the example they give here makes a lot of  sense, which is looking at data from surveys.  If you've got a whole bunch of like,  from one to five answers on surveys. 
Create Synthetic Centroids
All right, so let's take a look at this. And the way I like to build my algorithms is  
to create some, often, to create some synthetic  data that I know how I want it to behave.  And so we're going to create six clusters and  each class is going to have 750 samples in it. 
So first of all, I'm going to  randomly create six centroids. 
And so the centroid is going to be like  the middle of where my clusters are.  So I'm going to randomly create them. 
I need them n_clusters by 2 because I  need an X and a Y coordinate for each one. 
And so now I'm going to randomly  generate data around those six centroids. 
Okay, so to do that, I'm going to call a  little function I made here called sample. 
And I'm going to run it on  each of those six centroids. 
And so I'll show you what that looks like.  So here's what that data looks like. So the Xs are the six centroids  
and the colored dots is the data. So if you were given this data without the Xs,  
the idea would be to come back with  figuring out where the Xs would have been. 
Like where are these clustering around? And so if  you can get clusters, that's the goal here, is to  
find out that there's a few discreetly, distinctly  different types of data in your data set. 
So for example, for images, I've used this before  to discover that there are some images that look  
completely different to all the other ones. For example, they were taken at nighttime   or they're of a different  object or something like that. 
So how does sample work? Well  we're passing in the centroid  
and so what we want is we're going to get back…  So each of those centroids contains an X and a Y. 
So MultivariateNormal() is just like normal. It's going to give you back normally distributed   data, but more than one item. That's why it's multivariate. 
And so we passed in two means, a  mean for X and a mean for our Y. 
And so that's the mean that we're going to get. And our standard deviation is going to be five. 
Why do we use torch.diag( tensor([5.,  5.]) )? That's because we're saying,  
that's because that for multivariate  normal distributions, there's not   just one standard deviation for  each column that you get back. 
There could also be a connection between columns. So columns might not be independent. 
So you actually need, so it's called a  covariance matrix, not just a variance. 
We discussed that a little bit more in Lesson 9b,  if you're interested in learning more about that. 
Okay, so this is something that's going  to give us back random columns of data   with this mean and this standard deviation. And this is the number of samples that we want. 
And this is coming from PyTorch. So PyTorch has a whole bunch of   different distributions that you  can use, which can be very handy. 
So there's our data. Okay, so remember for clustering,  
we don't know the different colors. And we don't know where the Xs are.  That's kind of our job is to figure that out. We might just briefly also look at how to plot. 
So in this case, we want to plot  the Xs and we want to plot the data. 
So it looks like this. So all I do is I loop through each centroid. 
And I grab that centroid samples,  and they're just all done in order.  So I grab it from i time n_samples  up to (i +1) times n_samples. 
And then I create a scatterplot  with the samples on them.  And what I've done is I've created an axis here. And you'll see why later that we can also pass  
one in, but I'm not passing one in. So we create a plot and an axis.  And so in Matplotlib, you can keep  plotting things on the same axis. 
So then I plot on the centroid, a big  X, which is black, and then a smaller X,  
which is what is that, magenta. And so that's how I get these Xs.  So that's how plot data works. Okay, so how do we create something  
Mean shift algorithm
now that starts with all the dots  and returns where the Xs are?  
We're going to use a particular algorithm,  particular clustering algorithm called Mean Shift. 
And   Mean Shift is a nice clustering approach, because  you don't have to say how many clusters there are. 
So it's not that often that you actually  got to know how many clusters there are.  So we don't have to say. Quite a few things like the  
very popular k-means require you to say how many. Instead, we just have to pass them in a bandwidth,  
which we'll learn about, which can  actually be chosen automatically.  And it can also handle clusters of any shape. So they don't have to be ball-shaped  
like they are here. They can be kind of like   L-shaped or ellipse-shaped or whatever. And so here's what's going to happen. 
We're going to pick some point. So let's say we pick that point just there. 
And so what we now do is we  go through each data point. 
So we'll pick the first one. And so we then find the distance   between that point and every other point. So we're going to have to say,  
what is the distance between that  point and that point and that point and  
that point and that point and that point? And also  the ones further away, that point and that point. 
And you do it for every single point compared  to the one that we're currently looking at. 
Okay, so we get all of those as a big list. 
And now what we're going to do is we're going to  take a weighted average of all of those points. 
Now that's not interesting without the weighting. 
If we just take our average of all of  the points and how far away they are,   we're going to end up somewhere here, right?  This is the average of all the points. 
But the key is that we're  going to take an average. 
Let me find the right spot. The key is we need to find an average  
that is weighted by how far away things are. So for example, this one over here is a very  
long way away from our point of interest. And so it should have a very low weight in  
the weighted average, whereas this point  here, which is very close, should have  
a very high weight in our weighted average. So what we do is we create weights for every  
point compared to the one that we're  currently interested in using what's   called a Gaussian kernel that we'll look at. The key thing to know is that points that are  
further away from our point of interest, which  is this one, are going to have lower weights. 
That's what we mean there. They're penalized.  The rate at which weights fall to zero  is determined by this thing that we set  
at the start called the bandwidth. And that's going to be the standard   deviation of our Gaussian. So we take an average of  
all the points in the data set, a weighted  average weighted by how far away they are. 
So for our point of interest, this point's going  to get a big weight, this point's going to get  
a big weight, this point's going to get a big  weight, that point's going to get a tiny weight,  
that point's going to get an even tinier weight. So it's mainly going to be a weighted average  
of these points that are nearby. And the weighted average of those points,   I would guess, is going to be  somewhere around about here. 
And would have a similar thing for the  weighted average of the points near this one,   that's going to probably be somewhere  around about here, or maybe over here. 
And so it's going to move all of these points in  closer, it's almost like a gravity, right? They're  
kind of going to be moved like closer and closer  in towards this kind of gravitational center. 
And then these ones will go towards their  own gravitational center, and so forth. 
Okay, so let's take a look at it. 
All right, so what's the Gaussian kernel?  This is the Gaussian kernel, which was a sign  
in the original march for science, back in the  days when the idea of not following scientists  
was considered socially unacceptable. We used to have much for these things,   if you remember. So this is not normal. 
So this is the definition of the Gaussian kernel,  which is also known as the normal distribution.  This is the shape of it. I'm sure you've seen it before. 
And here is that formula copied  directly off the science march sign. 
Okay, here we are,   you can see the square root, two, pi, et cetera. Okay, and this here is the standard deviation. 
Plotting gaussian kernels
Now what does that look like? It's  very helpful to have something that   we can very quickly plot any function. That doesn't come with Matplotlib,  
but it's very easy to write one. Just say, oh, let's, as X, let's use all the   numbers from 0 to 10, 100 of them spaced evenly. That's what linspace() does. 
Linearly spaced 100 numbers in this range. That's going to be our Xs.  So plot those Xs and plot f(X), the Ys. So here's a very nice little plot_func we want. 
And here it is. And as you can see here,   we've now got something where if you are this,  like very close to the point of interest,  
you're going to get a very high weight. And if you're a long way away from the   point of interest, you'll get a very low weight. So that's the key thing that we wanted to remember  
is something that penalizes  further away points more. 
Now you'll notice here, I managed to plot   this function for a bandwidth of 2.5. And the way I did that was using this  
special thing from functools called partial. Now the first thing to point out here is  
that very often, drives me crazy, I see people  trying to find out what something is in Jupyter. 
And the way they do it is they'll scroll up  to the top of the notebook and search through   the imports and try to find it. That is the dumb way to do it. 
The smart way to do it is just  to type it and press shift enter,   and it'll tell you where it comes from. And you can get its help with question mark  
and you can get it source  code with two question marks.  Okay. So just type it to find out where it comes from. 
Okay. So this is,   as Siva's mentioned in the chat, also known  as currying or partial function application. 
This creates a new function. So let's just grab it. 
We create a new function and this function f  is, is the function Gaussian, but it's going  
to automatically pass bw equals 2.5. This is a partially applied function.  So I could type f(4), for example,  that's going to be a tensor. 
There we go. And you can see   that's exactly what this is. Go up to 4, go across.  Yep. About 0.44. 
So we use partial function  application all the time.  It's a very, very, very important tool. Without it, for example,  
plotting this function would have been more  complicated. With it, it was trivially easy. 
I guess the alternative, like one  alternative, which would be fine,   but slightly more clunky would be, we  could create a little function in line. 
So we could have said, Oh, plot a function  that I'm going to define right now,   which is called lamb—, which is  Lambda x, which is Gaussian of  
x with a bandwidth of 2.5. You could do that too.  You know, it's, it's fine. But, but yeah, partials I think are  
a bit neater, a bit less to think about. They often produce some neater and clearer code. 
Okay. Why did we decide to make the   bandwidth 2.5? As a, as a rule of thumb choose a  bandwidth, which covers about a third of the data. 
So if we kind of found ourselves somewhere  over here, right? A bandwidth which covers  
about a third of the data would be  enough to cover two clusters ish. 
So you'd want to be kind of like this big. So somewhere in the middle there  
so that's the basic idea. Yeah. 
So, but you can play around with bandwidths  and get different amounts of clusters. 
I should mention, like, often when you see  something that's kind of on the complicated side,   like a Gaussian, you can often simplify things. I think most of the implementations and writeups  
I've seen talk about using Gaussians,  but if you look at the shape of it,   it looks a lot like this shape. So this is a triangular weighting,  
which is just using clamp_min(). So it's just using a linear with clamp_min(). 
And yeah, it occurred to me that we  could probably use this just as well.  So I decided to define this triangular  weighting and then we can try both. 
Anyway, so we'll start with, we're  going to use the Gaussian version. 
All right.  So we're going to be literally moving all the  points towards their kind of center of gravity. 
So we don't want to mess up our original data. So we clone it.  That's a PyTorch thing is .clone(). It's very handy. 
And so big X is our matrix of data. I mean, it's actually a,  
that's right, matrix of data. Yeah.  And then little x will be our first point. And it's pretty common to use big X,  
capital letters for matrices. So this is our data. 
This is the first point. Okay.  So there it is. So we're going to start at (26.2, 26.3). 
So (26.2, 26.3). So somewhere up here. 
Calculating distances between points
So little x its shape is just,  it's a rank 1 tensor of shape 2,  
big X is a rank 2 tensor of 1,500  data points by 2, the x and y. 
And if we call x[None], that  would add a unit access to that. 
And the reason I'm going to show you  that is because we want to find the   distance from little x to everything in big X. And the way we do a distance is with minus,  
but you wouldn't be able to go,   you wouldn't be able to go x – X and get the  right actually do you get the right answer?  
Let's think about that extra shape. Oh, you've got that already. 
Oh no, actually that is going to work. Isn't it? So   yes. All right. 
So you can see why we've  got these two versions here.  If we do x[None], we've got  something of shape 1, 2. 
Now we can subtract that from something  of shape 1,500, 2, because the 2s match up  
because they're the same and the 1,500 and the 1  matches up because you remember our NumPy rules,  
everything matches up to a unit axis. So it's going to copy this  
matrix across every row of  this matrix and it works. 
But do you remember there's a special trick, which  is if you've got two shapes of different lengths,  
we can use the shorter length and  it's going to add unit axes to the   front to make it as long as necessary. So we actually don't need the x[None]. 
We can just use little x and it works because it's  going to say, is this compatible with this? Well,  
the last axis, remember we go right to left. The last access matches   the second last axis. Oh, it doesn't exist. 
So we pretend that there's a unit axis and it's  going to do exactly the same thing as this. 
So if you have not studied the broadcasting  from last week carefully, that might not  
have made a lot of sense to you. And so definitely at this point,   you might want to pause the video and go  back and reread the NumPy broadcasting  
rules from last time and practice  them because that's what we just did.  We used NumPy broadcasting rules and we're  going to be doing this dozens more times  
throughout the rest of the course and  many more times in fact in this lesson.  Okay. So now I think  
it's a pretty good place to have a pause. So I'll see you back here in nine minutes. 
Hi everybody. Welcome back.  So we had got to the point where we had managed  to get the distance between our first point x  
and all of the other points in the data. And so we're just looking at the   first eight of them here. So the very first distance  
is of course zero on the X axis and zero on  the Y axis because it is the first point. 
The other thing is that because we… the  way we created the clusters is they're all  
kind of next to each other in the list. So these are all in the first cluster.  So none of them are too far away from each other. So now that we've got all the distances,  
it's easy enough to, well not the distances on  X and Y, it's easy enough to get the distance  
… kind of Euclidean distance. So we can just square that  
difference and sum, and square root. And actually maybe this is a good time  
to talk about norms and to talk  about what we just did there. 
Calculating distances between points (illustrated)
We've got all these data points. 
So here's one of our data points and  here's the other one of our data points. 
And there's some distance across the X axis  and there's some distance along the Y axis. 
We could call that change in X and change in Y. 
And one way to think about this distance  then is it's this distance here. 
So to calculate that we can use Pythagoras. So a² + b² = c²  
Or in our case, so this would be c, a and b,  say, so in our case it would be the square root  
of the change in X squared  plus the change in Y squared. 
And rather than saying square root, we could say  
to the power of a half, another  way of saying the same thing.  But there's a different way  we could find the distance. 
We could first go along here and then go up here. 
And so that one would be change in X,  if you like, to the one plus change in  
Y to the one to the power of one-oneth. I'm writing it a slightly odd way for  
reasons you'll see in a moment. It's just this otherwise. 
In general, if we've got a whole  list of numbers, we can add them up. 
Let's say there are some  list V. We can add them up.  We can do each one to the  power of some number alpha. 
And take that sum to the one over alpha. And this thing here is called a norm. 
So you might remember we came across that last  week and we come across it again this week.  They basically come up, I don't know,  they might end up coming up every week. 
They come up all the time,  particularly because the two norm,   which we could write like this (||v||),  or we could write like this (|v|),  
or we could write like this (|v|2). They're all the two norm. 
This is just saying it's this equation   for alpha equals two. And Stefano is pointing out we  
should actually have an absolute value. I'm not going to worry about that.  We're just doing real numbers here. So we'll keep things simple.  Oh, well, I guess for higher than one. No, you're probably right. 
Something like three. Yeah, I guess we do need an absolute value there.  That's a good point because,  okay, we could have this one. 
And so the distance actually  has to be the absolute value.  So the change in X is the  absolute value of that distance. 
Yes, thank you, Stefano. Okay, so we'll have the absolute value.  Okay, so the two norm is what  happens when alpha equals two. 
And we would call this, in this case, we  would call this the Euclidean distance. 
But actually where it comes up more often   is when you're doing like a loss function. So the Mean Squared Error  
is just, well, the Root Mean Squared  Error, I should say, is just the two norm. 
Whereas the Mean Absolute Error is the one norm. And these are also known as L2 and L1 plus. 
And remember what we saw in that paper  last week, we saw it in this form,   there's a two up here, which is where  they got rid of the square root again. 
So that would have just been  
change in X squared plus change in Y squared. And now we don't even need the parentheses. 
Oopsie-Daisy. Okay.  So, all of this is to say that for,  you know, this comes up all the time,  
because we're very, very often interested in  distances and errors and things like that. 
I'm trying to think, I don't feel like I've  ever seen anything other than one or two. 
So although it is a general concept,  I don't think we're going to see   probably things other than  one or two in this course. 
I'd be excited if we do,  that would be kind of cool. 
So here we're taking the Euclidean  distance, which is the two norm. 
So this has got eight things in it,  because we've summed it over dimension one. 
So here's your first homework, is  to rewrite using torch.einsum(). 
You won't be able to get rid of the x minus  X, you'll still need to have that in there.  But when you've got a multiply followed by  a sum, you won't be able to get rid of the  
X square root either, you should be able  to get rid of the multiply and the sum by   doing it in a single torch.einsum(). So we're summing up over the first  
dimension, which is this dimension. So in other words, we're summing up   the X and the Y axis. Okay, so now  
Getting the weights and weighted average of all the points
we can get the weights by passing  those distances into our Gaussian. 
And so as we would expect, the  biggest weights, it gets up to 0.16. 
So the closest one is itself,  it's going to be at a big weight.  These other ones get reasonable weights and  the ones that are in totally different clusters  
have weights small enough that at three  significant figures they appear to be zero. 
Okay, so we've got our weights. So the weights are 1,500 long  
vector and of course our original data is  1,500 by 2, the X and the Y for each one. 
So we now want a weighted average. We want this data, we want  
its average weighted by this. So normally an average is the  
sum of your data divided by the count. That's a normal average. 
A weighted average, each item in your  data, let's put some i’s around here  
just to be more clear, each item in your  data is going to have a different weight. 
And so you multiply each one by the weights. And so rather than dividing by n,   which is just the sum of ones, we  would divide by the sum of weights. 
So this is an important concept to  be familiar with, weighted averages. 
So we need to multiply every  one of these x’s by this. 
Okay, so can we say weight times X? No. All right, why didn't that work?  
So, remember we go right to left. So first of all, it's going to say, let's  
look at the 2 and multiply that by the 1,500. Are they compatible? These are compatible if  
they're equal or if at least one of them is 1. These are not equal and they're not 1,  
so they're not compatible. That's why it says the size of a tensor a  
must match. Now when it says match,   it doesn't mean they have to be the same. One of them can be 1. 
Okay, that's what it means to match. They're either equal or one of them is 1.  So that doesn't work. On the other hand,  
what if this was 1,500 comma  1? If it was 1,500 comma 1,  
then they would match because the 1  and the 2 match because one of them   is a unit axis and the 1,500 and the  1,500 match because they're the same. 
So that's what we're going to do.  Because that would then copy this   to every one of these, which is what we want. We want weights for each of these X, Y tuples. 
So to add the trailing unit axis, we  say every row and a trailing unit axis. 
So that's what that shape looks like. So we can now multiply that by X. 
And as you can see, it's  now weighting each of them.  And so each of these x’s and y’s  down the bottom, they're all zero. 
So we can sum that up and then  divide by the sum of weights.  So let's now write a function  that puts all this together. 
So you can see this really important way of  like, to me, the only way that makes sense to  
do particularly scientific numerical programming,  I actually do all my programming this way,   but particularly scientific and numerical  programming is write it all out step by step,  
check every piece, have it all there documented  for you and for others, and then copy the cells,  
merge them together and indent them  to indent its control right spare  
bracket and put a function header on top. So here's all those things we just did. 
And now rather than just grabbing the  first x, we enumerate through all of them. 
So that's the distance we had before. That's the weight we had before.  There's the product we had before. And then finally sum across the rows,  
divide by the sum of the weights. So that's going to calculate  
for the ith, it's going to move. So it's actually changing capital X. 
So it's changing the ith thing in capital  X so that it's now the weighted sum. 
Oh, actually, sorry, the weighted average of all  of the other data, weighted by how far it is away. 
So that's going to do a single step. So the mean shift update is extremely   straightforward, which is clone the data,  iterate a few times, and do the update. 
So if we run it, takes 600 milliseconds.  And what I've done is I've plotted the centroids  moved by two pixels or two, well, not two pixels,  
two units so that you can see them. And so you can see the dots is where our data is. 
And they're dots now because every single data  point is on top of each other on a cluster.  And so you can see they are  now in the correct spots. 
So it has successfully clustered our data. So that's great news.  And so we could test out our hypothesis. Could we use triangular just as well  
as we could have used Gaussian? So  control slash comments and uncomments. 
Yep, we got exactly the same results. So that's good. 
It's really important to know these keyboard  shortcuts, hit H to get a list of them. 
Some things that are really important  don't have keyboard shortcuts.  So if you click help, edit keyboard shortcuts,  there's a list of all the things Jupyter can do. 
And you can add keyboard shortcuts  to things that don't have them.  So for example, I always add keyboard shortcuts  to run all cells above and run all cells below. 
As you can see, I type Q and then A  for above and Q and then B for below. 
All right. Now   that was kind of boring in a  way because it did five steps. 
But we just saw the result. What did it look like one step at a time?  
This isn't just fun. It's really important to be   able to see things happening one step at a time  because there are so many algorithms we do which  
are like updating weights or updating data. So for Stable Diffusion, for example,   you're very likely to want to show your  incrementally denoising and so forth. 
So in my opinion, it's important  to know how to do animations.  And I found the documentation for this  unnecessarily complicated because a lot of  
it's about how to make them performant. But most of the time we probably   don't care too much about that. So I want to show you a little trick, a simple  
way to create animations without any trouble. So Matplotlib.animation has something  
Matplotlib animations
called FuncAnimation. That's what we're going to use.  To create an animation, you  have to create a function. 
And the function, you're going to be calling  FuncAnimation, passing in the name of that  
function and saying how many times to run it. And that's what this frames argument.  This says run this function this many times. And then create an animation that basically  
contains the result of that with a 500  millisecond interval between each one. 
So what's this do one going to do? To create one  frame of animation, we will call our one_update(). 
Here it is. one_update().  We're going to call this. That's going to update our Xs. 
And then we're going to have an  axis, which we've created here. 
So we're going to clear whatever was on the  plot before and plot our new data on that axis. 
And then the only other thing you need to  do is that the very first time it calls it,   we want to plot it before running. 
And d is going to be passed  automatically the frame number.  So for the zeroth frame, we're  going to not do the update. 
We're just going to plot  the data as it is already. 
I guess another way we could have done  that would have been just to say if d, then  
do the update, I suppose.  That should work too. Maybe it's even simpler. 
Let's see if I just broke it. Okay.  So we're going to clone our data. We're going to create our figure in our subplots. 
We're going to call FuncAnimation()  calling do_one 5 times. 
And then we're going to display the animation.  And so let's see. So HTML takes some HTML and displays it. 
And to_jshtml() creates some HTML. That's why it's created this HTML   includes JavaScript. And so we'll click run  
one, two, three, four, five. There's the five steps.  So if I click loop, you'll see  them running again and again. 
Fantastic.  So that's how easy it is to  create a Matplotlib animation. 
So hopefully now you can use that to play around  with some fun Stable Diffusion animations as well. 
You don't just have to use to_jshtml(). You can also create movies, for example. 
You can call to_html5_video()  would be another option and   you can save an animation as a movie file. So there's all these different options for that,  
but hopefully that's enough to get you started. So  
for your homework, I would like you when you  create your K-means or whatever, to try to create  
your own animation or create an animation of some  Stable Diffusion thing that you're playing with. 
So don't forget this important ax.clear().  Without the ax.clear() it prints it on top of   the last one, which sometimes is what you want, to  be fair, but in this case, it's not what I wanted. 
All right. So kind of slow,   half a second for not that much data. 
Accelerating our work by putting it on the GPU
I'm sure it would be nice if it was faster. Well, the good news is we can GPU accelerate it. 
The bad news is it's not going to GPU  accelerate that well because of this loop. 
This is looping 1,500 times. If we, so looping is not going to run on the GPU. 
So the best we could do with this  would be to move all this to the GPU.  Now the problem is that calling something on the  GPU 1,500 times from Python is a really bad idea  
because there's this kind of huge communication  overhead of this kind of flow of control and data  
switching back between the CPU and the GPU. It's the kernel launching overhead. 
It's bad news. So you don't want to have   a really big, fast Python loop that inside  it calls CUDA code because GPU code. 
So we need to make all of this run without  the loop, which we could do with broadcasting. 
So let's roll up our sleeves and try to  get the broadcast version of this working. 
So generally speaking, the way we  tend to do things with broadcasting   on a GPU is we create batches or mini batches. 
So to create batches or mini batches, we  know we just call them batches nowadays.  We create a batch size. So let's say we're  
going to do a batch size of five. So we're going to do five at a time. 
All right. So how do we do five at a   time? This is only doing one at a time. How do we do five at a time?  
As before, let's clone our data. And this time little x for our testing.  So we're going to do everything ahead  of time, little tests as we always do. 
This is not now X[0] anymore, but it's X[:bs]. So it's the first five. 
This is now the first five items. Okay.  So little x is now a 5 by 2 matrix. This is our mini batch, the first five items. 
As before our data itself is 1,500 by 2. All right.  So we need a distance calculation, but  previously our distance calculation.  
Previously our distance calculation only  worked if little x was a single number  
and it returned just the distances from  that to everything in big X, but we need  
something that's actually going to be, return a  matrix, right? We've got, let's say we've got 5  
by 2 in little x and then in big X, we've got  something much bigger, not to scale obviously. 
We've got 1500 by 2. And what is the distance between  
these two things? Well, if you think about  it, there's going to be a distance between  
item one and item one, but there's also going  to be a distance between item one and item two,  
and there's going to be a distance between,  let's use a different color for the next one,  
item two and item one.  Right? So the output of this is  actually going to be a matrix. 
The distances are actually going to give us a  matrix where, I mean, it doesn't matter which  
way around we do it. We can decide,   but if we do it this way around, for each of the  5 things in the mini batch, there will be 1,500  
distances, the distance between every one.  So we're going to need to do  broadcasting to do this calculation. 
So  
this is the function that we're going to create,  and it's going to create this, as you can see,  
5 by 1,500 output, but let's see how we get it. 
So can we do X – x? No, we can't. Why is that? That's because big  
X is 1,500 by 2 and little x is 5 by 2. So it's going to look at, remember our rules,  
right to left. Are these compatible? Yes, they are. 
They're the same. These compatible?  
No, they're not. Okay, because they're different.  So that's not possible to do. What if, though, we wanted to…  
what if we insert in big X  an axis at the start here  
and in little x, we add an axis in the  middle here, then now these are compatible  
because you've got, they're the same  —because I should use arrows really. 
These are compatible because one of them is a 1.   And these are compatible because  one of them is the 1 as well. 
So they are all compatible. And what it's going to do is it's going to do this  
subtraction between these directly. And it's going to copy this across all 1,500 rows. 
It'll copy it. This is going to be copied.  And then this table across 5 rows, and then  this will be copied across these 1,500 rows. 
Because that's what broadcasting does.  I mean, it's not really copying,  but it's effectively copying. 
And so that gives us, we can now subtract them,   and that gives us what we  wanted, which is 5 by 1,500. 
And there's also by 2 because  there's both the X and the Y.  So that's why this works. That's what this is doing here. 
It's taking this attraction, it's squaring them  and then summing over that last shortest axis,  
summing over the X and the Y squareds. And then take square root.  I don't know why I said torch dot square root. We could have just put dot square  
root at the end, but same, same. In fact, it's worth mentioning that.  So most things that you can do on tensors,  you can either write torch dot as a function,  
or you can write it as a method. Generally speaking, both should be fine. 
Not everything, but most  things work in both spots.  Okay, so now we've got this  matrix, which is 5 by 1,500. 
And the nice thing is that our  Gaussian kernel doesn't actually   have to be changed to get the  weights, believe it or not. 
And the reason for that is —now how do we get  the source code? I could move back up there,   or I can just type gaussian?? and see it. And the nice thing is that this is just,  
this is a scalar. So it broadcasts over anything.  And then this is also just a scalar. So this is all going to work fine without  
any fiddling around. Okay. 
So now we've got a five by 1,500 weight. So that's the weight for each of the 5 things,  
our mini batch, for each of the 1,500  things, each of the most compared to.  And then we've got the shape of the data  itself, X dot shape, which is the 1,500 points. 
So now we want to apply each one of  these weights to each of these columns. 
So we need to add a unit axis to the end. So to add a unit axis to the end,  
we could say [:, :, None], but dot dot dot  ( [...] ) means all of the axes up until  
however many you need. So in this case,   the last one comma none ( […, None] ). This is going to add an axis to the end. 
So this is going to turn, this is going to turn   weight dot shape from 5 comma  1,500 to 5 comma 1500 comma 1. 
And this is going to add an axis to the start. Remember it's the same as X[None, :, :]  
And so let's check our rules. Left, right to left. These are compatible because one of them is 1. 
These are compatible because they're both the same  and these are compatible because one of them is 1. 
Okay. So it's going to be copying each weight across  
to each of the X and Y, which is what we want. We want to, we want to weight both of those  
components and it's going to  copy each of the 1,500 points. 
Sorry. Each of the point 5 times,   because we do in fact want to weight every one of  the 5 things in our mini batch, a separate set of  
weights for each of them. So that sounds perfect.  So that's how I think through these calculations. 
Okay. So we can now do that multiplication, which  
is going to give us something of 5 by 1,500 by 2,  because we end up with the maximum of our ranks. 
And then we sum up over those 1,500 points and  that's going to give us, now, 5 new data points. 
Now something that you might notice here  is that we've got a product and a sum. 
And when you see a product and a sum that  tells you that maybe we should use einsum. 
So in this case, we've got our weight. We've got 5 by 1,500. 
So let's call those i and  j as for the 5 and 1,500.  We've got the X is 1500 by 2. Now we want to take the product of that and that. 
So we'll need to use the same name for this row. So we use j again. 
Okay. And then k   is the number of rows. That's the 2. 
And then we want to end up with i by k. So  torch.einsum() gives exactly the same result. 
That's great. But you might recognize this.  That's exactly the same einsum we had just  before when we were doing matrix multiplication. 
Oh, that is a matrix multiplication. We've just reinvented matrix multiplication  
using this rather nifty trick. So we could also just use that. 
And so, again, this is like what I was just  playing around with this morning as I started  
to look at this and I was thinking like, oh, can  we simplify this? I don't like this kind of like  
messing around with axes and  summing over dimensions and whatnot.  And so it's nice to get things down to einsum or,  better still, get them down to matrix multipliers. 
It's just clearer. It's stuff that we all recognize   because we use them all the time. They all work. 
Performance would be pretty similar, I suspect. Okay. 
So now that we've got that,  we then need to do our sum. 
And we've got our 5 points. This is our 5 denominators. 
So we've got our numerator that we  calculated up here for our weighted average. 
The denominator is just the  sum of the weights, remember.  And so numerator divided by  denominator is our answer. 
So again, we've gone through every step.  We've checked out all the  dimensions all along the way. 
So nothing's going to surprise us. Don't try and write a function like   this just bang from scratch, right?  You've got to drive yourself crazy. 
Instead do it step by step. So here's our mean shift algorithm.  Clone the data. Go through five iterations. 
And now go from naught to n, batch size at a time. 
So Python has something called slices. So we can create a slice  
of X starting at 1 up to i plus batch size, right?  Unless you've gone past n, which goes use n. 
And so then we just copying and pasting each  of the lines of code that we had before.  Actually, I just copy the cells and merge them. 
Of course, I don't actually copy and  paste because it's slow and boring.  And there's my final step to create the new X[s]. And so notice here, s is not a single thing. 
It's a slice of things. You might not have seen   slice before, but this is just internally  what Python's doing when you use colon. 
And it's very convenient when you need  to use the same slice multiple times. 
Okay.  So let's do that using CUDA. I would run it first without CUDA,  
but I mean, I've done all the  steps before, so it should be fine.  So pop it on the GPU and run meanshift(). And let's see how long that takes. 
It takes 1 millisecond. And previously without GPU,   it took 400 milliseconds. And you know, the other thing we should probably  
think about doing is looking at other batch  sizes as well, because now we're looping over  
batches, right? So if we make the batch size  bigger, that for loop is going to do less looping. 
So what if we make that 16? Will that be any  faster? I actually never tried this before. 
That's interesting.  It's actually slower. Huh. 
There you go. Fascinating.  What if it was eight? Amazing. So the big batches don't quite seem to  
be working so well for some reason. So I wonder if I've...  Hang on. What's going on? Why is it  
changing how it should be? My batch size was 5. Why is it slower suddenly?  
I think it's just a bit varying. That's probably the answer.  So it just varies a lot. Okay. 
So it doesn't seem like changing the  batch size is changing much here. 
So that's fine. So we'll just leave it where it was.  And then check looking at the data. Oh, that looks lovely. 
Oh, I see. Thank you people on YouTube   pointing out that I'm passing batch size. So I actually need to put it here. 
Right. So if we used a batch size of 5,   no wonder it was messing up. Oh, look at that.  I've totally made it slow now. 157 milliseconds. 
Ha ha. Okay.  64.  13 milliseconds. All right. 
Finally, that makes much more sense. 256. 
1024. Okay.  So the bigger, bigger is better. 
And I guess we could actually  do all 5,000 at once probably.  Nice. 
All right. Thank you YouTube friends  
for solving that bizarre mystery. 
Okay. All right.  So that's pretty great. I mean, you know, to see  
that we can GPU optimize a mean shift.  Actually Googled for this to  see if it's been done before. 
And it's the kind of thing that  people like write papers about. 
So I think it's great that we can do it so easily  with PyTorch, which is the kind of thing that  
previously had been considered, you know, a  very challenging academic problem to solve. 
So maybe you can do something  similar with some of these.  Now I haven't told you what these are. So part of the homework is to go read  
about them and learn about them. DBSCAN, funnily enough, actually is   an algorithm that I accidentally invented and then  discovered a year later had already been invented. 
That was a long time ago. I was playing around with J, which is the   successor to APL on a very old Windows phone. And I had a long plane flight and I came up  
with an algorithm and implemented  the whole thing on my phone using J   and then discovered a year later  that I just invented DBSCAN. 
This is actually a really cool algorithm and  it's got a lot of similarities to Mean Shift. 
LSH comes up all the time. So that's great.  And in fact, I have a strong feeling, and  I've been thinking about this for a while,  
that something like LSH could be used  to speed this whole thing up a lot.  Because if you think about it, and again,  maybe this already exists, I don't know. 
But if you think about it, when we did that  distance calculation, the vast majority  
of the weights are nearly zero. And so it seems pointless to create  
that big, you know, kind of eventually  1,500 by 1,500 matrix, that's slow. 
It would be much better if we just found  the ones that were like pretty close by   and just took their average. And so you want an optimized  
nearest neighbors, basically. And so this is an example of  
something that can give you a kind of  a fast nearest neighbors algorithm. 
Or, you know, there are things like k-d  trees and octrees and stuff like that.  So if you want to like, have a bonus bonus,  
invent a new mean shift algorithm, which picks  only the closest points to avoid quadratic  
time. All right.  So not very often you get an assignment,  which is to invent a new mean shift algorithm,  
I guess a super super bonus. Super super bonus. 
Publish a paper that describes it. 
All right, you definitely get  four points if you do that.  We'll give you a number of points equal to  the impact factor of the journal you get it  
published in. Okay. 
So what I want to do now   is move on to calculus, which for some  of us may not be our favorite topic. 
That's funny, Stefano wrote the  einsum version here already.  I didn't notice. Okay.  Always ahead of his time, that guy. Let's talk about calculus. 
If you're not super comfortable with derivatives  and what they are and why we care, 3Blue1Brown  
has a wonderful series called the essence of  calculus, which I strongly recommend watching. 
It's just a pleasure, actually,  to watch as it's everything that   is on 3Blue1Brown, a pleasure to watch. And so we're not going to get into backprop today. 
Instead, we're just going to  have a quick chat about calculus. 
Calculus refresher
Where do we start? So the good news is,  just like you don't have to know much  
linear algebra at all, you basically just  need to know about matrix multiplication. 
You also don't need to know much calculus at all. Just derivatives. 
So let's think about what derivatives are. So I'm going to borrow actually the same  
starting point that 3Blue1Brown uses in  one of their videos to consider a car. 
And we're going to see how far away  from home it is at various time points. 
Okay. So after a minute, let's say after a second,  
it's traveled 5 meters. 
And then after 2 seconds,   it's traveled 10 meters. Okay. 
And after 3 seconds, you can probably  guess, it's traveled 15 meters. 
So there's this concept here of  a, got it the wrong way around,  
obviously. So time, distance. 
Okay. So there's this concept of location. 
It's like how far have you traveled  at a particular point in time? So we  
can look at one of these points and  find out how far that car has gone. 
We could also take two points and we can say where  did it start at the start of those two points  
and where did it finish at  the end of those two points. 
And we can say between those  two points, how much time passed   and how far did they travel in 2  seconds, they traveled 10 meters. 
So we could now also say, all right, well,  the slope of something is rise over run. 
Oopsie-Daisy, 10 meters in 2 seconds. And notice we don't just divide the numbers. 
We also divide the units. We get 5 meters per second. 
So, this here is now change  the dimensions entirely. 
We're now not looking at distance, but  we're looking at speed or velocity. 
And it's equal to rise over run. It's equal to the rate of change. 
And what it says, really, is as time,  the X axis, goes up by 1 second,  
what happens to the distance in  meters? As one second passes, how does  
the number of meters change?   And so maybe these aren't points at all. Maybe there's a function. 
Right? It's a continuum of points. And so you can do that for the function. 
So the function is a function of time. Distance is a function of time. 
And so we could say, what's  the slope of that function?  
And we can get the slope from point  A to point B using rise over run. 
So from t1 to t2, the amount of  time that's passed is t2 minus t1. 
That's how much time has passed. Let's say this is t1, this is t2. 
And the distance that they've  traveled, well, they've moved from  
wherever they are at the end to  wherever they were at the start. 
So that's the change in distance  divided by the change in time.  Change in distance divided by change in time. 
Okay. Let's say that's Y. 
So another way, now the thing is, when we talk  about calculus, we talk about finding a slope. 
But we talk about finding a slope of something,  
that's often more tricky than this,  right? We have slopes of things  
that look more like this. And we say,   what's this slope? Oops, I'm terrible at drawing. Let's maybe put it over here because  
I'm left-handed. What's this slope?  
Now what does it mean to  have the idea of a velocity  
at an exact moment in time?  It doesn't mean anything. 
At an exact moment in time, you're just like,  it's frozen, right? What's happening exactly  
now? But what you can do is you can say, well,  what's the change in time between a bit before  
our point and a bit after our point? And what's  the change in distance between a bit before our  
point and a bit after our point? And so you  can do the same kind of rise over run thing,  
right? But you can make that distance between  t2 and t1 smaller and smaller and smaller. 
So let's rewrite this in a slightly different way. Let's call the denominator the distance between t1  
plus a little bit, we'll call  it d. It's that minus t1. 
So this is t2, right? It's t1 plus a little bit. So we say, oh, here's t1, let's add a little bit. 
And notice that when we write it this way,  let's actually let's do the rest of it.  So now f(t2) becomes F( t1 plus a little bit ). And this is the same. 
And now notice here that t1 plus  d minus t1, we can delete all that  
because it just comes out to d. 
So this is another way of calculating  the slope of our function.  And as d gets smaller and smaller  and smaller, we're kind of getting  
a triangle that's tinier and tinier and tinier. And it still makes sense, it's still that some  
time has passed and the car has moved, right? But  it's just smaller and smaller amounts of time. 
Now if you did calculus at college or  at school, you might've done all this  
stuff messing around with limits and  epsilon, delta and blah, blah, blah. 
I've got really good news. It turns out   you can actually just think of this d as a  really small number, where d is the difference. 
And so when we calculate the slope,  
we can write it in a slightly different way  as the change in Y divided by the change in X. 
This here is the change in Y and  this here is the change in X. 
And so in other words, this here is a  very small number, a very small number. 
And this here is the result in the function  of changing by that very small number. 
And this way of thinking about calculus is  known as the calculus of infinitesimals. 
And it's how Leibniz originally developed it. And it's been turned into a whole theory nowadays. 
And the reason I talk about it here  is because when we do calculus,  
you'll see me doing stuff all the time where  I act like dX is a really small number. 
And when I was at school, I was  told I wasn't allowed to do that.  I've since learned that it's  totally fine to do that. 
So for example, next lesson, we're going to be  looking at the chain rule, which looks like this. 
So dY/dX equals dY/dU times dU/dX.  
And I'm just going to say, oh, these  two small numbers can cancel out. 
And that's why obviously they're the same thing. And that's all going to work out nicely. 
So anywho, what would be very helpful  would be if before the next lesson,  
if you're not totally up to date with your  remembering all the stuff you did in high school  
about calculus, is watch the 3Blue1Brown course.  We are not going to be looking, I  don't think at all, at integration. 
So you don't have to worry about that. Also we are not going to, on the whole,  
be doing any derivatives by hand. So for example, there are rules such as  
dY/dX if Y equals X squared is 2X. 
These kind of rules you're not  really going to have to learn   because PyTorch is going to do them all for you. The one that we care about is going to be  
the chain rule, but we're going  to learn about that next time.  Okay, I hope I don't get beaten to a  bloody pulp the next time I walk into  
a mathematician's conference. I suspect I might,   but hopefully I get away with this. 
I think it's safe. We'll see how we go. 
Thanks everybody very much for joining me and  really look forward to seeing you next time where  
we're going to do back propagation from scratch. We've already learned to multiply matrices,  
so once we've got back propagation as well,  we'll be ready to train a neural network. 
All right, thanks all. Bye.

---

# 13

Hi everybody and welcome to Lesson 13, where  we're gonna start talking about back propagation.  
Before we do, I'll just mention that there  was some great success amongst the folks in   the class during the week on working with  flexing their tensor manipulation muscles.
So far the fastest mean shift algorithm, which  has a similar accuracy to the one I displayed,  
is one that actually randomly chooses data points,  
a subset. And I actually think that's a great  approach. Very often random sampling and random  
projections are two excellent ways of speeding  up algorithms. So it'd be interesting to see  
if anybody during the rest of the course comes  up with anything faster than random sampling.
Also been seeing some good Einstein  summation examples and implementations   and continuing to see lots of  good DiffEdit implementations.  
So congratulations to all the students. And I hope  those of you following along the videos in the  
MOOC will be working on the same homework as well  and sharing your results on the fast.ai forums.
So now we're gonna take a  look at notebook number three  
in the normal repo, course22p1  repo (*course22p2*).  
And we're going to be looking at  the forward and backward passes of  
a simple Multi-Layer Perceptron, a neural network. The initial stuff up here is just importing things  and just settings and stuff that just copying  
and pasting some stuff from previous notebooks  around paths and parameters and stuff like that. So we'll skip over this. So we'll often be  kind of copying and pasting stuff from one  
notebook to another's kind of first cell  to get things set up. And I'm also loading  
in our data for MNIST as tensors. Okay, so  we, to start with, need to create the basic  
architecture of our neural network. And I did mention at the start of  
the course that we will briefly review  everything that we need to cover. So   we should briefly review what basic neural  networks are and why they are what they are.
Linear models & rectified lines (ReLU) diagram
So to start with, let's consider a linear  model. Oops, that's not how I do it.  
So let's start by considering a linear model of,  well, let's take the most simple example possible,  
which is we're gonna pick a single pixel from  our MNIST pictures. And so that will be our X.  
And for our Y values, then we'll have some  loss function of how good is this model.  
Sorry, not some loss function. Let's  create it even simpler. For our Y value,  
we're going to be looking at how likely is it  that this is, say, the number three, based on  
the value of this one pixel. So the pixel is going  to be the number three, the pixel. So the pixel,   its value will be X and the probability of  being the number three, we'll call Y. And  
if we just have a linear model,  then it's gonna look like this.
And so in this case, it's saying that the brighter  this pixel is, the more likely it is that it's the  
number three. And so there's a few problems with  this. The first one obviously is that as a linear  
model, it's very limiting because maybe, you know,  we actually are trying to draw something that  
looks more like this. So how would you do that?  Well, there's actually a neat trick we can use  
to do that. What we could do is, well, let's first  of all talk about something we can't do. Something  
we can't do is to add a bunch of additional  lines. So consider what happens if we say, okay,  
well, let's add a few different lines. So let's  also add this line. So what would be the sum of  
our two lines? Well, the answer is, of  course, that the sum of the two lines   will itself be a line. So it's not gonna help  us at all match the actual curve that we want.
So here's the trick. Instead, we  could create a line like this.  
That actually,  
we could create this line. And now  consider what happens if we add this   original line with this new, what's not  a line, right? It's two line segments.
So what we would get is this, everything to the  left of this point is going to not be changed  
if I add these two lines together, because this  is zero all the way, and everything to the right   of it is going to be reduced. It looks like  they've got similar slopes. So we might end up  
with, instead, so this would all disappear here.  And instead, we would end up with something like  
this. And then we could do that again, right?  We could add an additional line that looks a  
bit like that. So it would go, but this time it  could go even further out here. And it could be  
something like this, see. So what if we added that? Well, again, at  the point underneath here, it's always zero,  
so it won't do anything at all. But after that,  it's going to make it even more negatively sloped.  
And if you can see, using this approach, we could  add up lots of these rectified lines, these lines  
that truncate at zero, and we could create any  shape we want with enough of them. And these  
lines are very easy to create because actually all  we need to do is to create just a regular line,  
just create a regular line, right?  Which we can move up, down, left, right,  
change its angle, whatever. And then just say, if it's greater than zero,  truncate it to zero. Or we could do the opposite  
for a line going the opposite direction. If it's  less than zero, we could say truncate it at zero.  
And that would get rid of, as we want,  this whole section here, and make it flat.
Okay, so these are rectified lines.   And so we can sum up a bunch of these together  to basically match any arbitrary curve.  
So let's start by doing that. Oh,  the other thing we should mention,   of course, is that we're going  to have not just one pixel,  
but we're going to have lots of  pixels. So to start with the, kind of,  
most, you know, slightly, the only  slightly less simple approach,  
we could have something where we've got, you  know, pixel number one and pixel number two.   We're looking at two different pixels to see  how likely they are to be the number three.  
And so, that would allow us to draw  
more complex shapes that have  some kind of surface between them.
Okay, and then we can do exactly the same thing,   is to create these surfaces, we can add  up lots of these rectified lines together,  
but now they're going to be kind of rectified  planes. But it's going to be exactly the same  
thing. We're going to be adding together a bunch  of lines, each one of which is truncated at zero.
Okay, so that's the quick  review. And so to do that,  
Multi Layer Perceptron (MLP) from scratch
we'll start out by just defining a few variables.  So n is the number of training examples, m is the  
number of pixels, c is the number  of possible values of our digits.
And so here they are 50,000 samples,  784 pixels and 10 possible outputs.  
Okay, so what we do is to, is we basically  decide ahead of time how many of these  
line segment thingies to add up. And so the number that we create   in a layer is called the number  of hidden nodes or activations.
So we'll call that nh. So let's just  arbitrarily decide on creating 50 of those.  
So in order to create lots of lines, which  we're then going to truncate at zero,  
we can do a matrix multiplication.  So with a matrix multiplication,  
we're going to have something  where we've got 50,000 rows  
by 700, was it 784? Yeah, by 784 columns.  
And we're going to multiply that by  something with 784 rows and 10 columns.
And why is that? Well, that's because if we take  this very first line of this first vector here,  
row one, we have 784 values. They're the pixel  values of the first image. Okay, so this is  
our first image. And so they're each going to  each of those 784 values, but we multiply it by  
each of these 784 values in the first column,  the zero index column. And that's going to  
give us a number in our output. So our output  is going to be 50,000, 50,000 images by 10.
And so that result, we'll multiply those  together and we'll add them up. And that   result is going to end up over here in this  first cell. And so each of these columns,  
it's going to eventually represent, if  this is a linear model, in this case,  
this is just the example of doing a linear model,  each of these cells is going to represent the   probability. So this first column will be the  probability of being a 0 and the second column  
will be the probability of 1. The third column  will be the probability of being a 2 and so forth. So that's why we're going to have these 10  columns, each one allowing us to weight the  
784 inputs. Now, of course, we're going to do  something a bit more tricky than that, obviously  
we're going to have a 784 by 50 input going into  a 784 by 50 output to create the 50 hidden layers. 
Then we're going to truncate those at  zero and then multiply that by a 50 by   10 to create our 10 output.  So we'll do it in two steps.  
So the way SGD works is we start with  just, this is our weight matrix here.  
And this is our data. This is  our outputs. The way it works  
is that this weight matrix is initially filled with random values.   We also call this contains our pixel  values, this contains the results.
So W is going to start with random values.  
So here's our weight matrix. It's going to have,  as we discussed, 50,000 by 50 random values.  
And it's not enough just to multiply, we  also have to add. So that's what makes it  
a linear function. So we call those the biases,  the things we add. We can just start those at  
zeros. So we'll need one for each output,  so 50 of those. And so that'll be layer one.
And then as we just mentioned, layer two will  be a matrix that goes from 50 hidden. And now  
I'm going to do something totally cheating  to simplify some of the calculations for   the calculus. I'm only going to create one  output. Why am I going to create one output?  
That's because I'm not going to use cross  entropy just yet. Instead, I'm going to use MSE.  
So actually I'm going to create one output,  
which will literally just be what  number do I think it is from 0 to 10.  
And so then we're going to compare those to  the actual, so these will be our y predictors.   We normally use a little hat for that, and  we're going to compare that to our actuals.  
And yeah, in this very hacky approach,  let's say we predict over here the number  
9 and the actual is the number 2, and  we'll compare those together using MSE,  
which will be a stupid way to do it because it's  saying that 9 is further away from being 2 than 2,  
9 is further away from 2 than it is from 4 in  terms of how correct it is, which is not what  
we want at all, but this is what we're going  to do just to simplify our starting point. So that's why we're going to have a single  output for this weight matrix and a single  
output for this bias. So a linear, let's  create a function for putting x through  
a linear layer with these weights and these  biases. So it's a matrix multiply and an add.
All right, so we can now try it. So if we multiply  our x, oh, we're doing x_valid this time. So just  
to clarify, x_valid is 10,000 by 784. So if we  put x_valid through our weights and biases with  
a linear layer, we end up with a 10,000 by  50. So 10,000, 50 long hidden activations.  
They're not quite ready yet because  we have to put them through ReLU.   And so we're going to clamp at zero. So  everything under zero will become zero.  
And so here's what it looks like when we go  through the linear layer and then the ReLU.   And you can see here's a tensor with a bunch of  things, some of which is zero or they're positive.  
And so that's the result of  this matrix multiplication. Okay, so to create our basic MLP, multilayer  perceptron from scratch, we will take our  
mini batch of x's, xb is a x batch. We will  create our first layer's output with a linear,  
and then we will put that through a ReLU,  and then that will go through the second   linear. So the first one uses the w1, b1, okay,  these ones. And the second one uses the w2, b2.  
And so we've now got a simple model. And as  we hoped, when we pass in the validation set,  
we get back 10,000 digits, so 10,000  by 1. Great, so that's a good start.
Okay, so let's use our ridiculous loss  function of MSE. So, our results is 10,000 by 1  
Loss function from scratch - Mean Squared Error (MSE)
and our x_valid, sorry, our y_valid is just  a vector. Now what's gonna happen if I do  
res minus y_valid? So before you continue in  the video, have a think about that. What's  
gonna happen if I do res minus y_valid by thinking  about the NumPy broadcasting rules we've learned?
Okay, let's try it. Oh, terrible. We've ended up  with a 10,000 by 10,000 matrix. So a 100 million  
points. Now we would expect an MSE to just  contain 1,000 points. Why did that happen?
The reason it happened is because we  have to start out at the last dimension  
and go right to left. And we compare the  10,000 to the 1 and say, are they compatible?  
And the answer is, that's right, Alexi in  the chat's got it right, broadcasting rules. So the answer is that this one will  be broadcast over these 10,000.  
So this pair here will give us 10,000  outputs. And then we'll move to  
the next one. And we'll also move here to the next  one. Oh, oh!, there is no next one. What happens?  
Now, if you remember the rules, it inserts a unit  axis for us. So we now have 10,000 by 1. So that  
means each of the 10,000 outputs from here will  end up being broadcast across the 10,000 rows  
here. So that means that we'll end up for each  of those 10,000, we'll have another 10,000. So   we'll end up with a 10,000 by 10,000 output. So  that's not what we want. So how could we fix that?
Well, what we really would  want, we want this to be  
10,000, 1 here. If that was 10,000, 1, then we'd  compare these two, right to left, and they're  
both 1. So those match, and there's nothing to  broadcast because they're the same. And then  
we'll go to the next one, 10,000 to 10,000, those  match. So they just go element wise for those. And  
we'd end up with exactly what we want. We'd have  ended up with 10,000 results. Or, alternatively,  
we could remove this dimension. And then again,  same thing. We're then gonna add right to left  
compatible 10,000. So they'll  get element wise operation.  
So in this case, I got rid of the trailing comma  one. There's a couple of ways you could do that.
One is just to say, okay, grab  every row and the zeroth column   of res, and that's gonna turn it  from a 10,000 by 1 into a 10,000.  
Or alternatively, we can say  .squeeze(). Now .squeeze() removes all  
trailing unit vectors, and possibly also  prefix unit vectors. I can't quite recall.  
I guess we should try. So, let's  say res[None, :, None], q.shape.  
Okay, so if I go q.squeeze().shape. Okay,  so all the unit vectors get removed. 22:11.400 --> 22:12.400 So I'm gonna do that.
22:12.400 --> 22:14.400 And then I'm gonna do the same thing. 22:14.400 --> 22:16.400 So I'm gonna do the same thing. 22:16.400 --> 22:18.400 So I'm gonna do the same thing. 22:18.400 --> 22:24.400 Okay, so all the unit vectors get removed.
Sorry, all the unit dimensions  get removed, I should say. Okay, so now that we've got a way to remove that  axis that we didn't want, we can use it. And if  
we do the subtraction, now we get 10,000,  just like we wanted. So now let's get our  
training and validation wise. We'll turn  them into floats because we're using MSE.   So let's calculate our predictions for the  training set, which is 50,000 by 1. And so  
if we create an MSE function that just does what  we just said we wanted, so it does the subtraction  
and then squares it and then  takes the mean, that's MSE.  
So there we go. We now have a loss  function being applied to our training set.
Gradients and backpropagation diagram
Okay. Now we need gradients. So  as we briefly discussed last time…  
gradients are slopes. And in fact, maybe it  would even be easier to look at last time.  
So this was last time's notebook. And so  we saw how… the gradient at this point is…  
is the slope here.   And so it's the, as we discussed, right?  The, as we discussed, rise over run.  
Now, so that means as we increase,  in this case, time by one,  
the distance increases by how  much? That's what the slope is.  
So why is this interesting?  
The reason it's interesting is because, let's  consider our neural network. Our neural network  
is some function that takes two things, two  groups of things. It contains a matrix of our  
inputs. And it contains our  weight matrix. And… we want to…  
and let's assume we're also putting it  through a loss function. So let's say,   well, I mean, I guess we can be explicit  about that. So we could say we then take  
the result of that and we put it through some  loss function. So these are the predictions.  
And we compare it to our… to our actual  dependent variable. So that's our neural net.  
And that's our loss function.
Okay. So if we can get the derivative of the loss…  
with respect to, let's say, one particular  weight, so let's say weight number zero,  
what is that doing? Well, it's saying as  I increase the weight by a little bit,  
what happens to the loss? And if it says,  oh, well, that would make the loss go down,  
then obviously I want to increase the  weight by a little bit. And if it says,  
oh, it makes the loss go up, then  obviously I want to do the opposite.   So the derivative of the loss with respect to the  weights, each one of those, tells us how to change  
the weights. And so to remind you, we then change  each weight by that derivative times a little bit  
and subtract it from the original weights. And we  do that a bunch of times, and that's called SGD.
Now, there's something interesting going  on here, which is that in this case,  
there's a single input and a single output. And  so the derivative is a single number at any point.  
It's the speed. In this case, the vehicle's  going. But consider a more complex function…  
like, say, this one. Now, in this case,  there's one output, but there's two inputs.  
And so if we want to take the  derivative of this function,  
then we actually need to say, well, what happens  if we increase X by a little bit, and also what  
happens if we increase Y by a little bit? And in  each case, what happens to Z? And so in that case,  
the derivative is actually going to contain  two numbers, right? It's going to contain the  
derivative of Z with respect to Y, and it's going  to contain the derivative of Z with respect to X.
What happens if we change each of these two  numbers? So, for example, these could be, as we   discussed, two different weights in our neural  network, and Z could be our loss, for example.  
Now, we've got actually 784 inputs, right?  So we would actually have 784 of these. So  
we don't normally write them all like that. We  would just say, use this little squiggly symbol  
to say the derivative of the loss across all of  them with respect to all of the weights. Okay,  
and that's just saying that there's a whole bunch  of them. It's a shorthand way of writing this.
Okay, so it gets more complicated still  though because think about what happens  
if, for example, you're in the first layer where  we've got a weight matrix that's going to end up  
giving us 50 outputs, right? So for every image,  we're going to have 784 inputs to our function,  
and we're going to have 50 outputs  to our function. And so in that case,  
I can't even draw it, right? Because like for  every, even if I had two inputs and two outputs,  
then as I increase my first input, I'd actually  need to say, how does that change both of the  
two outputs? And as I change my second input,  how does that change both of my two outputs?  
So for the full thing, you actually are going  to end up with a matrix of derivatives. It  
basically says for every input that you change, by  a little bit, how much does it change every output  
of that function? So you're  going to end up with a matrix.   So that's what we're going to be doing is we're  going to be calculating these derivatives,  
but rather than being single numbers,  they're going to actually contain  
matrices with a row for every input and a  column for every output. And a single cell  
in that matrix will tell us as I change this input  by a little bit, how does it change this output?
Now, eventually, we will  end up with a single number  
for every input, and that's because our loss  in the end is going to be a single number. And  
this is like a requirement that you'll find when  you try to use SGD is that your loss has to be a  
single number. And so we generally get it by doing  the sum or a mean or something like that. But as  
you'll see on the way there, we're going to have  to be dealing with these matrix of derivatives.
So I just want to mention,  
as I might have said before,  I can't even remember,  
Matrix calculus resources
there is this paper that Terrence Parr and I  wrote a while ago, which goes through all this.  
And it basically assumes that you only know high  school calculus, and if you don't, check out Khan  
Academy, but then it describes matrix calculus  in those terms. So it's going to explain to you  
exactly, and it works through lots and lots of  examples. So for example, as it mentions here,  
when you have this matrix of derivatives, we  call that a Jacobian matrix. So there's all  
these words. It doesn't matter too much if you  know them or not, but it's convenient to be able  
to talk about, you know, the matrix of all of the  derivatives if somebody just says the Jacobian.
It's a little convenience. It's a little bit  easier than saying the matrix of all of the  
derivatives where all of the rows are things  that are all the inputs and all the columns are   the outputs. So yeah, if you want to really  understand, get to a point where papers are  
easier to read in particular, it's quite useful  to know this notation and definitions of words.  
You can certainly get away without  it. It's just something to consider.
Okay, so we need to be able to calculate  derivatives at least of a single variable, and  
I am not going to worry too much  about that, a) because that is   something you do in high school math, and  b) because your computer can do it for you.  
And so you can do it symbolically using  something called SimPy, which is really  
Gradients and backpropagation code
great. So if you create two symbols called  X and Y, you can say please differentiate  
X squared with respect to X, and if you do  that, SimPy will tell you the answer is 2X.
If you say differentiate 3X² + 9 with respect  to X, SimPy will tell you that's 6X. And  
a lot of you probably will have used Wolfram Alpha  that does something very similar. I kind of quite  
like this because I can quickly do it inside my  notebook and include it in my prose. So I think  
SimPy is pretty cool. So basically, yeah, you  can quickly calculate derivatives on a computer.
Having said that, I do want to talk  about why the derivative of 3X² + 9  
equals 6X because that's  going to be very important.  
So 3X² + 9. So we're going to start with the  information that the derivative of A to the B  
with respect to A equals B times A. So  for example, the derivative of X squared  
with respect to X equals 2X. So that's just  something I'm hoping you'll remember from  
high school or refresh your memory using Khan  Academy or similar. So there that is there.
So what we could now do is we could  rewrite this derivative as 3u + 9.  
And then we'll write u = X². Okay, now  this is getting easier. The derivative of  
two things being added together is  simply the sum of their derivatives.  
Oh, forgot B minus 1 in the exponent. Thank  you. Sorry. B A to the power of B minus 1.  
That's what it should be. Which would be 2X  to the power of 1, and the 1 is not needed.  
Thank you for fixing that. All right. So we just sum them up. So we get the  derivative of 3U is actually just,  
well, it's going to be the derivative of  that plus the derivative of that. Now,  
the derivative of any constant with respect to  a variable is 0 because if I change something,  
an input, it doesn't change the constant.  It's always 9. So that's going to end up as 0.  
And so we're going to end up with  dY/dU equals something plus 0,  
and the derivative of 3U with respect to U is just  3 because it's just a line. So that's its slope.
Okay, but that's not dY/dX. We want to do up  to dY/dX. Well, the cool thing is that dY/dX  
is actually just equal to dY/dU dU/dX.  
So I'll explain why in a moment, but for now  then let's recognize we've got dY… sorry dU/dX.  
We know that one, 2X. So we can now  multiply these two bits together.  
And we will end up with 2X times 3, which is  6X, which is what SimPy told us. So fantastic.
Okay, this is something we need to know  really well, and it's called the chain rule.   And it's best to understand it intuitively.  
So to understand it intuitively, we're going  to take a look at an interactive animation.  
So I found this nice interactive  animation on this page here,  
webspace.ship.edu/msrenault/geogebracalculus/derivative_intuitive_chain_rule.html
Okay, and the idea here is that  we've got a wheel spinning around,  
Chain rule visualized + how it applies
and each time it spins around, this is X  going up. Okay, so at the moment, there's some  
change in X, dX over a period of time. All  right, now, this wheel is eight times bigger  
than this wheel. So each time this goes  round once, if we connect the two together,  
this wheel would be going round  four times faster, because  
the difference between, the multiple between 8  and 2 is 4. Maybe I'll bring this up to here.
So now that this wheel has got twice as big a  circumference as the u-Wheel, each time this  
goes round once, this is going round two times.  So the change in U, each time X goes round once,  
the change in U will be 2. So that's what dU/dX is  saying. The change in U for each change in X is 2.  
Now, we could make this interesting by  connecting this wheel to this wheel.
Now, this wheel is twice as small as this wheel.  
So now we can see that, again, each time this  spins round once, this spins round twice,  
because this has twice the circumference  of this, so therefore, dY/dU equals 2.
39:57.400 --> 40:00.400 But now that means every   time this goes round once, this goes round twice,  every time this one goes round once, this gun goes  
round twice. So therefore, every time this one  goes round once, this one goes round four times.  
So dY/dX equals 4. So you  can see here how the two,  
well, how the dU/dX has to be multiplied with the  dY/dU to get the total. So this is what's going on  
in the chain rule. And this is what you  want to be thinking about, is this idea that  
you've got one function that  is kind of this intermediary.   And so you have to multiply the two impacts to  get the impact of the X wheel on the Y wheel.  
So I hope you find that useful. I find this,  personally, I find this intuition quite useful.
So why do we care about this? Well, the reason we  care about this is because we want to calculate  
the gradient of our MSE applied to our model.  And so our inputs are going through a linear,  
they're going through a ReLU, they're  going through another linear, and then   they're going through an MSE. So there's  four different steps going on. And so we're  
going to have to combine those all together.  And so we can do that with the chain rule.
So if our steps are that  
loss function is, so we've got the loss function,  which is some function of the predictions  
and the actuals, and then we've got,  the second layer is a function of,  
actually, let's say, let's call this the output  of the second layer. Slightly weird notation,  
but hopefully it's not too bad. It’s going  to be a function of the ReLU activations.  
And the ReLU activations are a function  of the first layer, and the first layer  
is a function of the inputs. Oh, and of  course, this also has weights and biases.  
So we're basically going to have to  calculate the derivative of that.  
Okay, but then remember that this is itself a  function. So then we'll need to multiply that  
derivative by the derivative of that. But that's  also a function, so we have to multiply that 
derivative by this. But that's also a function,  so we have to multiply that derivative by this. So that's going to be our approach.  We're going to start at the end.  
We're going to take its derivative, and then we're  going to gradually keep multiplying as we go each  
step through. And this is called back propagation.  So back propagation sounds pretty fancy,  
but it's actually just using the chain rule. Gosh,  I didn't spell that very well, prop-a-gation.  
It's just using the chain rule, and as  you'll see, it's also just taking advantage   of a computational trick of memorizing  some things on the way. And in our chat,  
Siva made a very good point about understanding  nonlinear functions in this case, which is just  
to consider that the wheels could be growing  and shrinking all the time as they're moving.  
But you're still going to have the same compound  effect, which I really like that. Thank you, Siva.
There's also a question in the chat about why  is this colon comma zero being placed in the  
function, given that we can do it outside the  function. Well, the point is we want an MSE   function that will apply to any output. We're not  using it once. We want it to work any time. So we  
haven't actually modified preds or anything  like that or y_train. So we want this to be  
able to apply to anything without us having to  preprocess it, it’s basically the idea here.
Okay. So let's take a look at the basic  idea. So here's going to do a forward pass  
and a backward pass. So the forward pass is  where we calculate the loss. So the loss is,  
oh, I've got an error here. That  should be diff. There we go.
So the loss is going to be the output of our  neural net minus our target squared and then  
take the mean. Okay. And then our output is going  to be the output of the second linear layer.  
The second linear layer's input will be  the ReLU. The ReLU's input will be the   first layer. So we're going to take our  input, put it through a linear layer,  
put that through a ReLU, put that through  a linear layer and calculate the MSE.
Okay. That bit hopefully  is pretty straightforward.   So what about the backward  pass? So the backward pass,  
what I'm going to do, and you'll see why  in a moment, is I'm going to store the   gradients of each layer. So, for example, the  gradients of the loss with respect to its inputs  
in the layer itself. So I'm going to create a  new attribute. I could call it anything I like.   I'm just going to call it .g. So I'm going to  create a new layer, a new attribute called out.g,  
which is going to contain the gradients. You  don't have to do it this way, but as you'll see,   it turns out pretty convenient. So that's  just going to be two times the difference,  
because we've got difference squared.  So that's just the derivative. And then  
we have taken the mean here, so we have to do  the same thing here, divided by the input shape.  
And so that's those gradients. That's good.  And now what we need to do is multiply  
by the gradients of the previous layer. So  here's our previous layer. So what are the   gradients of a linear layer? I've  created a function for that here.  
So the gradient of a linear layer, we're going to  need to know the weights of the layer. We're going  
to need to know the biases of the layer. And then  we're also going to know the input to the linear  
layer because that's the thing that's actually  being manipulated here. And then we're also going  
to need the output because we have to multiply by  the gradients because we've got the chain rule.
So again, we're going to store the gradients of  our input. So this would be the gradients of our  
output with respect to the input. And that's  simply the weights, because the weights, so a  
matrix multiplier is just a whole bunch of linear  functions. So each one slope is just its weight.  
But you have to multiply it by the gradient  of the outputs because of the chain rule.  
And then the gradient of the outputs with  respect to the weights is going to be the  
input times the output summed up.
I'll talk more about that in a moment. The  derivatives of the bias is very straightforward.  
It's the gradients of the output added  together because the bias is just a constant  
value. And so for the chain rule, we simply  just use output times one, which is output.  
So for this one here, again, we have to  do the same thing we've been doing before,   which is multiply by the output gradients  because of the chain rule. And then we've  
got the input weights. So every single one  of those has to be multiplied by the outputs.  
And so that's why we have to  do an unsqueeze minus one. So what I'm going to do now is I'm going to show  you how I would experiment with this code in order  
Using Python’s built in debugger
to understand it, and I would encourage you to do  the same thing. It's a little harder to do this  
one cell by cell because we kind of want to put it  all into this function like this. So we need a way  
to explore the calculations interactively. And the  way we do that is by using the Python debugger.
Here is how you, let me see a few ways to do  this. Here's one way to use the Python debugger.  
The Python debugger is called  pdb. So if you say pdb.set_trace()  
in your code, then that tells the debugger to stop  execution when it reaches this line. So it sets  
a breakpoint. So if I call forward and backward,  you can see here it's stopped and the interactive  
Python debugger, ipdb, has popped up with an arrow  pointing at the line of code it's about to run.
And at this point, there's a whole range of things  we can do to find out what they are. We hit h   for help. Understanding how to  use the Python debugger is one  
of the most powerful things I think  you can do to improve your coding.
So one of the most useful things you can  do is to print something. You see all   these single letter things? They're  just shortcuts. But in a debugger,  
you want to be able to do things quickly.  So instead of typing print, I just type p.   So for example, let's take a look at the shape  of the input. So I type p for print, input.shape.  
So I've got a 50,000 by 50 input to the  last layer. So that makes sense. These   are the hidden activations coming into the  last layer for every one of our images.
What about the output gradients?  
And there's that as well.  And actually a little trick,   you can ignore that. You don't have to  use the p at all if your variable name  
is not the same as any of these commands.  So I could have just typed out.g.shape.  
Get the same thing. Okay. So you can also put in expressions.  So let's have a look at the shape of this.  
So the output of this is, let's see if it makes  sense. We've got the input, 50,000 by 50. We put  
a new axis on the end, unsqueeze(-1) is the same  as doing dot, as indexing it with [..., None].
Let's put a new axis at the end. So that  would have become 50,000 by 50 by 1.  
And then the out.g.unsqueeze() we're putting in  the first dimension. So we're going to have 50,000  
by 50 by 1 times 50,000 by 1 by 1. And so we're  only going to end up getting this broadcasting  
happening over these last two dimensions,  which is why we end up with 50,000 by 50 by 1.  
And then with summing up, this makes sense,  right? We want to sum up over all of the inputs.  
Each image is individually contributing to the  derivative. And so we want to add them all up  
to find their total impact, because remember  the sum of a bunch of, the derivative of the   sum of functions is the sum of the derivatives  of the functions. So we can just sum them up.
Now, this is one of these situations where if  you see a times and a sum and an unsqueeze, it's  
not a bad idea to think about Einstein summation  notation. Maybe there's a way to simplify this.  
So first of all, let's just see how we can do some  more stuff in the debugger. I'm going to continue.  
So just continue running. So press c for continue,  and it keeps running until it comes back again to  
the same spot. And the reason we've come to the  same spot twice is because lin_grad() is called  
two times. So we would expect that the second  time we're going to get a different bunch of  
inputs and outputs. And so I can print out a  tuple of the inputs and output gradient. So now,  
yeah, so this is the first layer going into the  second layer. So that's exactly what we'd expect.
To find out what called this function,  you just type w. w is where am I? And so  
you can see here where am I? Oh,  forward and backward was called.   See the arrow? That called lin_grad() the  second time, and now we're here in w.g =.
If we want to find out what w.g ends up  being equal to, I can press n to say go  
to the next line. And so now we've moved from  line five to line six. So the instruction point  
is now looking at line six. So I could  now print out, for example, w.g.shape,  
and there's the shape of our weights.
One person on the chat has pointed out  that you can use breakpoint instead of this  
import pdb business. Unfortunately, the breakpoint  keyword doesn't currently work in Jupyter or in  
IPython, so we actually can't, sadly. That's why  I'm doing it the old-fashioned way. So this way,  
maybe they'll fix the bug at some point,  but for now, we have to type all this.
Okay, so those are a few things to know about,  but I would definitely suggest looking up a   Python pdb tutorial to become very familiar with  this incredibly powerful tool because it really is  
so very handy. So if I just press continue again,  it keeps running all the way to the end, and it's  
now finished running forward and backward.  So when it's finished, we would find that  
there will now be, for example, a w1.g because  this is the gradient that it just calculated,  
and there would also be a x_train.g and so forth.
Okay, so let's see if we can simplify this  a little bit. So I would be inclined to take  
these out and give them their own variable names  just to make life a bit easier. It would have  
been better if I'd actually done this before the  debugging, so it would be a bit easier to type.  
So let's set i and o equal to input  and output dot g dot unsqueeze.  
Okay, so we'll get rid of our breakpoint and  double check that we've got our gradients.
Okay.  
And I guess before we rerun it, we  should probably set those to zero.  
What I would do here to try things  out is I'd put my breakpoint there,  
and then I would try things. So let's go next. And  so I realize here that what we're actually doing  
is we're basically doing exactly the same thing  as an einsum would do. So I could test that out  
by trying an einsum, right?, because  I've just got this as being replicated,  
and then I'm summing over that dimension,  because that's the multiplication that I'm   doing. So I'm basically multiplying the first  dimension of each and then summing over that  
dimension. So I could try running that,  and, ah, it works. So that's interesting.  
And I've got zeros because I did  x_train dot zero. That was silly.  
So that should be dot gradients dot zero. Okay. So let's try doing an einsum.  
And there we go. That seems to be working.  That's pretty cool. So we've multiplied this  
repeating index. So we were just multiplying the  first dimensions together and then summing over  
them. So there's no i here. Now, that's not quite  the same thing as a matrix multiplication, but we  
could turn it into the same thing as a matrix  multiplication just by swapping i and j so that  
they're the other way around. And that way we'd  have ‘ji, ik’. And we can swap into dimensions  
very easily. That's what's called the transpose.  So that would become a matrix multiplication if  
we just use the transpose. And in NumPy, the  transpose is the capital T attribute. So here  
is exactly the same thing using a matrix  multiply and a transpose. And let's check.  
Yeah, that's the same thing as well. Okay, cool. So that tells us that  now we've checked in our debugger  
that we can actually replace  all this with a matrix multiply.  
We don't need that anymore. Let's see if it works.  
It does. All right. x_train.g. Cool.
Okay, so hopefully that's convinced  you that the debugger is a really handy   thing for playing around with numeric  programming ideas or coding in general.  
And so I think now's a good time to take a break.  So let's take a eight-minute break, and I'll see  
you back here. Actually, seven-minute break. I'll  see you back here in seven minutes. Thank you.
Okay, welcome back. So we've  calculated our derivatives,  
and we want to test them. Luckily, PyTorch  already has derivatives implemented,  
so I've got to totally cheat and use PyTorch to  calculate the same derivatives. So don't worry  
about how this works yet, because we're  actually going to be doing all this from   scratch anyway. For now, I'm just going  to run it all through PyTorch and check  
that their derivatives are the same as ours,  and they are, so we're on the right track. Okay, so this is all pretty  clunky. I think we can all agree,  
and obviously it's clunkier than what we do  in PyTorch. So how do we simplify things?   There's some really cool  refactoring that we can do.  
Refactoring the code
So what we're going to do is we're going to create  a whole class for each of our functions, for the   ReLU function and for the linear function. So the  way that we're going to do this is we're going to  
create a Dunder call. What does Dunder call  do? Let me show you. So if I create a class,  
and we're just going to set that to print hello.  
So if I create an instance of that class,  and then I call it as if it was a function,  
oops, missing the Dunder bit here,   call it as if it's a function, it  says hi. So in other words, you know,  
everything can be changed in Python. You can  change how a class behaves. You can make it look,  
work like a function, and to do that, you simply  define Dunder call. You could pass it an argument,  
like so. Okay, so that's what Dunder call  does. It just says it's just a  
little bit of syntax sugary kind of  stuff to say, I want to be able to   treat it as if it's a function without any method  at all. You can still do it the method way. You  
could have done this. Don't know why you'd  want to, but you can. Because it's got this  
special magic named Dunder call, you don't  have to write the dot Dunder call at all. So here, if we create an instance of the ReLU  class, we can treat it as a function. And what  
it's going to do is it's going to take its input  and do the ReLU on it. But if you look back at the  
forward and backward, there's something very  interesting about the backward pass, which 
is that it has to know about, for example, this  intermediate calculation gets passed over here.  
This intermediate calculation gets passed over  here. Because of the chain rule, we're going to  
need some of the intermediate calculations, and  not just the chain rule, because of actually how  
the derivatives are calculated. So we need to  actually store each of the layer intermediate  
calculations. And so that's why ReLU doesn't  just calculate and return the output, but it  
also stores its output, and it also stores its  input. So that way, then, when we call backward,  
we know how to calculate that. We set the inputs  gradient, because remember, we stored the input,  
so we can do that, right? And it's going to  just be, oh, input greater than zero dot float,  
right? So that's the definition,  okay, of the derivative of a ReLU.  
And then chain rule. So that's how we can  calculate the forward pass and the backward pass  
for ReLU, and we're not going to have to then  store all this intermediate stuff separately.   It's going to happen automatically. So we can do  the same thing for a linear layer. Now, a linear  
layer needs some additional state, weights and  biases. ReLU doesn't, right? So there's no in it.  
So when we create a linear layer, we have  to say, what are its weights, what are its   biases? We store them away, and then when we  call it on the forward pass, just like before,  
we store the input. So that's exactly the same  line here. And just like before, we calculate the  
output and store it and then return it, okay?  And this time, of course, we just call lin.
And then for the backward pass, it's the  same thing, okay? So the input gradients  
we calculate just like before. Oh, .t()  is exactly the same with a little t as  
big T is as a property. So that's the  same thing. That's just the transpose.  
Calculate the gradients of the weights.  Again, with a chain rule and the bias,  
just like we did it before, and they're  all being stored in the appropriate places.
And then for MSE, we can do the same thing. We  don't just calculate the MSE, but we also store   it. And we also, now the MSE needs two things, an  input and a target, so we'll store those as well.  
So then in the backward pass, we can calculate  its gradient of the input as being two times.  
The difference. And there it all is. Okay. So our model now is much easier to define.  We can just create a bunch of layers, linear, w1,  
b1, ReLU, linear, w2, b2. And then we can store  an instance of the MSE. So this is not calling  
Mse. It's creating an instance of the Mse class.  And this is an instance of the Lin class. This is  
an instance of the Relu class. So they're just  being stored. So then when we call the model,  
we pass it our inputs and our target. We go  through each layer, set x equal to the result  
of calling that layer, and then pass that  to the loss. So there's something kind of   interesting here that you might have noticed,  which is that we don't have… There we do it.
Something interesting here is that we don't have  two separate functions inside our model, the loss  
function being applied to a separate neural net.  But we've actually integrated the loss function  
directly into the neural net, into the model. See  how the loss is being calculated inside the model?  
Now, that's neither better nor worse than  having it separately. It's just different.   And so generally a lot of HuggingFace stuff does  it this way. They actually put the loss inside  
the forward. Most stuff in fastai and a lot of  other libraries does it separately, which is the  
loss is a whole separate function, and the model  only returns the result of putting it through   the layers. So for this model, we're going to  actually do the loss function inside the model.
So for backward, we just do each thing. So  self.loss.backward(). So that self.loss is the  
Mse() object. So that's going to call backward,  right? And it's stored when it was called here.  
It was storing, remember, the inputs, the targets,  the outputs, so it can calculate the backward().  
And then we go through each layer is in  reverse, right? This is back propagation,   backwards reversed, calling backward on each  one. So that's pretty interesting, I think.
So now we can calculate the  model. We can calculate the loss.  
We can call backward. And then we can check that  each of the gradients that we stored earlier  
are equal to each of our new gradients.
Okay, so William's asked a very good question.  That is, if you do put the loss inside here,  
how on earth do you actually get predictions?  So generally what happens is in practice,  
HuggingFace models do something like this. They'll  say self.preds equals x. And then they'll say  
self.final_loss equals that. And then  return self.final_loss. And that way,  
I guess you don't even need that last bit. Well,  that's with them anyway. That is what they do. So   we'll leave it there. And so that way you can  kind of check, like, model.preds, for example.
So it'll be something like that. Or  alternatively, you can return not just the loss,   but both as a dictionary, stuff like that. So  there's a few different ways you could do it.  
Actually, now I think about it, I think that's  what they actually return both as a dictionary.   So it would be like return dictionary loss  equals that, comma, preds equals that,  
something like that, I guess is what they would  do. Anyway, there's a few different ways to do it.
Okay, so hopefully you can see that this is  really making it nice and easy for us to do our  
forward pass and our backward pass without all  of this manual fiddling around. Every class now  
can be totally separately considered and can  be combined however we want. We could create  
layers. So you could try creating  a bigger neural net if you want to. But we can refactor it more. So basically, as  a rule of thumb, when you see repeated code,  
self.inp equals inp, self.inp equals inp, self.out  equals return self.out, self.out equals return  
self.out. That's a sign you can refactor things.  And so what we can do is a simple refactoring is  
to create a new class called module. And module  is going to do those things we just said. It's   going to store the inputs. And it's going to call  something called self.forward in order to create  
our self.out because remember that was one of  the things we had again and again and again,   self.out, self.out. And then return it. And so  now there's going to be a thing called forward, 
which actually in this it doesn't do anything   because the whole purpose of  this module is to be inherited.
When we call backward, it's going to call  self.backward passing in self.out because  
notice all of our backwards always wanted to  get hold of self.out, right? Self.out, self.out,  
because we need it for the chain rule. So let's  pass that in and pass in those arguments that  
we stored earlier. And so star means take all  of the arguments regardless whether it's zero,  
one, two or more and put them into a list. And  then that's what happens when it's inside the  
actual signature. And then when you call  a function using star, it says take this   list and expand them into separate arguments  calling backward with each one separately.
So now for Relu, look how much simpler it  is. Let's copy the old Relu to the new Relu.  
So the old Relu had to do all this  storing stuff manually. Handed out  
all the self.stuff as well. But now we can get  rid of all of that and just implement forward  
because that's the thing that's being called  and that's the thing that we need to implement.  
And so now the forward of Relu just does the  one thing we want, which also makes the code   much cleaner and more understandable. Ditto for  backward. It just does the one thing we want.  
So that's nice. Now we still have to multiply  it by the chain rule manually.  
But so the same thing for linear (*Lin), same  thing for Mse. So these all look a lot nicer. And  
one thing to point out here is that there's often  opportunities to manually speed things up when you  
create custom autograd functions in PyTorch.  And here's an example. Look, this calculation  
is being done twice, which seems like  a waste, doesn't it? So at the cost of  
some memory, we could instead store  that calculation as diff. Right?
And I guess we'd have to store it for use  later, so it would need to be self.diff.  
And at the cost of that memory,  
we could now remove this redundant calculation  because we've done it once before already and  
stored it and just use it directly. And this is  something that you can often do in neural nets. So  
there's this compromise between storing things,  the memory use of that, and then the computational  
speedup of not having to recalculate it.  This is something we come across a lot.   And so now we can call it in the same way, create  our model, passing in all of those layers. So you  
can see with our model, we're just, so the model  hasn't changed at this point. The definition was  
up here. We just pass in the layers. Sorry,  not the layers, the weights for the layers.  
Calculate the loss, call backward,  and look, it's the same. Hooray.
Okay. So thankfully PyTorch has  written all this for us. And remember,  
according to rules of our game,  once we've reimplemented it,   we're allowed to use PyTorch's version.  So PyTorch calls their version nn.Module.  
And so it's exactly the same. You  inherit from nn.Module. So if we   want to create a linear layer just like this  one rather than inheriting from our module,  
we will inherit from their module.  But everything's exactly the same.   So we create our, we can create our random  numbers. So in this case, rather than passing  
in the already randomized weights, we're actually  going to generate the random weights ourselves and  
the zeroed biases. And then here's our linear  layer, which you could also use Lin for that,  
of course. So we defined our forward. And  why don't we need to define backward? Because  
PyTorch already knows the derivatives of all  of the functions in PyTorch, and it knows how  
to use the chain rule. So we don't have to  do the backward at all. It'll actually do   that entirely for us, which is very cool. So  we only need forward. We don't need backward.
So let's create a model that uses nn.Module.  Otherwise, it's exactly the same as before.  
And now we're going to use PyTorch's mse_loss()  because we've already implemented ourselves.  
It's very common to use torch.nn.functional as F.  This is where lots of these handy functions live,  
including mse_loss(). And so now you  know why we need the colon, comma, none,  
because you saw the problem if we don't have  it. And so create the model, call backward.  
And remember, we stored our gradients in  something called .g. PyTorch stores them  
in something called .grad, but it's doing exactly  the same thing. So there is the exact same values.
So let's take stock of where we're up to.  So we've created a matrix multiplication  
from scratch. We've created linear layers. We've  created a complete backprop system of modules we  
can now calculate both the forward pass and the  backward pass for linear layers and values so  
we can create a multilayer perceptron. So we're  now up to a point where we can train a model.
So let's do that.
minibatch training, notebook number four.  So same first cell as before. We won't go  
through it. This cell's also the same as  before, so we won't go through it. Here's  
the same model that we had before, so we won't  go through it. So just rerunning all that to see.
Okay. So the first thing we should do, I  think, is to improve our loss function so  
it's not total rubbish anymore. So if  you watched Part 1, you might recall  
that there are some Excel notebooks. One of  those Excel notebooks is entropy example.
Okay. So this is what we looked at. So just to  remind you, what we're doing now is we're saying,  
okay, rather than outputting a single number  for each image, we're going to instead output  
ten numbers for each image. And so  that's going to be a one-hot encoded  
set of, it will be like 1, 0, 0, 0, et cetera.  And so then that's going to be, well, actually  
the outputs won't be 1, 0, 0. They'll be basically  probabilities, won't they? So it'll be like 0.99,  
0.01, et cetera. And the targets will  be one-hot encoded. So if it's the digit  
0, for example, it might be 1, 0, 0, 0, 0,  dot, dot, dot for all the ten possibilities.  
And so to see how good is it, so in this case  it's very good. It had a 0.99 probability  
prediction that it's a zero and indeed it is  because this is the one-hot encoded version.
And so the way we implement that is we don't  even need to actually do the one-hot encoding,  
thanks to some tricks. We can actually just  directly store the integer, but we can treat   it as if it's one-hot encoded. So we can just  store the actual target zero as an integer.  
So the way we do that is we say, for example,  for a single output, oh, it could be cat,  
let's say cat, dog, plane, fish, building.  The neural net spits out a bunch of outputs.  
What we do for softmax is we go e to the power of  each of those outputs. We sum up all of those e  
to the power of ‘s. So here's the e to the power  of each of those outputs. Here's the sum of them.  
And then we divide each one by the sum. So divide  each one by the sum. That gives us our softmaxs.  
And then for the loss function, we then compare  those softmaxs to the one-hot encoded version.  
So let's say it was dog. Then it's going to  have a one for dog and zero everywhere else.  
And then softmax, this is from this nice blog post  here. This is the calculation sum of the ones and  
zeros. So each of the ones and zeros multiplied  by the log of the probabilities. So here is the  
log probability times the actuals. And  since the actuals are either 0 or 1 and  
only one of them is going to be a 1, we're  only going to end up with one value here.   And so if we add the up, it's  all 0 except for one of them. 
So that's cross entropy. So in this special  case where the output's one-hot encoded,  
then doing the one-hot encoded multiplied  by the log softmax is actually identical  
to simply saying, oh, dog is in this row.  Let's just look it up directly and take its  
log softmax. We can just index directly  into it. So it's exactly the same thing.
So that's just review. So if you  haven't seen that before, then yeah,   go and watch the Part 1 video where we  went into that in a lot more detail.
Okay. So here's our softmax calculation.  It's e to the power of each output divided  
by the sum of them, or we can use sigma  notation to say exactly the same thing.  
And as you can see, Jupyter Notebook lets us  use LaTeX. If you haven't used LaTeX before,  
it's actually surprisingly easy to learn. You just  put dollar signs around your equations like this,  
and your equations' backslash is going to  be kind of like your functions, if you like,  
and curly parentheses, curlies are used to  kind of for arguments. So you can see here,  
here is e to the power of, and then underscore is  used for a subscript. So this is x subscript i,  
and power of is used for  superscripts. So here's dots.  
You can see here it is, dots. So it's actually,  yeah, learning LaTeX is easier than you might  
expect. It can be quite convenient for  writing these functions when you want to. So anyway, that's what softmax is. As we'll see in  a moment, well, actually, as you've already seen,  
in cross-entropy, we don't really want softmax,  we want log of softmax. So log of softmax is,  
here it is. So we've got x.exp(), so e  to the x, divided by x dot exp dot sum,  
and we're going to sum up over the last dimension.  And then we actually want to keep that dimension  
so that when we do the divided by, we want a  trailing unit axis for exactly the same reason  
we saw when we did our MSE loss function. So if  you sum with keepdim=True, it leaves a unit axis  
in that last position so we don't have to put it  back to avoid that horrible out of product issue.
So this is the equivalent of this and then .log().  
So that's log of softmax. So there is the  log of the softmax with the predictions.  
Now, in terms of high school math that you  may have forgotten, but you definitely are  
going to want to know, a key piece in that  list of things is log and exponent rules.  
So check out Khan Academy or similar if  you've forgotten them, but a quick reminder  
is, for example, the one that we mentioned here. log(a/b) = log(a) - log(b)
And equivalently, log(a * b) = log(a) + log(b)
And these are very handy because, for  example, division can take a long time,  
multiplier can create really big numbers that  have lots of floating point error. Being able  
to replace these things with pluses and minuses is  very handy indeed. In fact, I used to give people  
an interview question 20 years ago at a company  which I did a lot of stuff with SQL and math.
SQL actually only has a sum function for group  by clauses, and I used to ask people how you  
would deal with calculating a compound interest  column where the answer is basically that you  
have to say, because this compound interest is  taking products, so it has to be the sum of the   log of the column and then e to the power  of all that. So there's like all kinds of  
little places that these things come in handy,  but they come into neural nets all the time.
So we're going to take advantage of that because  we've got a divided by that's being logged.  
And also, rather handily, we're going to  have, therefore, the log of x.exp() minus  
the log of this, but exp and log are opposites,  so that is going to end up just being x minus.  
So log softmax is just x minus all this logged.  And here it is, all this logged. So that's nice.
So here's our simplified version. Okay,  now there's another very cool trick,  
which is one of these things I figured out  myself and then discovered other people had   known it for years. So not my trick, but  it's always nice to rediscover things.  
The trick is what's written here.  Let me explain what's going on.   This piece here, the log of this sum, right,  this sum here, we've got x det exp dot sum. Now,  
x could be some pretty big numbers, and e to the  power of that is going to be really big numbers.   And e to the power of things creating really big  numbers, well, really big numbers, there's much  
less precision in your computer's floating point  handling. The further you get away from zero,  
basically. So we don't want really big numbers,  particularly because we're going to be taking   derivatives. And so if you're in an area that's  not very precise, as far as floating point math  
is concerned, then the derivatives are going to  be a disaster. They might even be zero because   you've got two numbers that the computer can't  even recognize as different. So this is bad.
But there's a nice trick we can do to make it a  lot better. What we can do is we can calculate the  
max of a, sorry, the max of x, right?, and we'll  call that a. And so then rather than doing the log  
of the sum of e to the xi, we're instead going to  define a as being the minimum, sorry, the maximum  
of all of our x values. It's our  biggest number. Now, if we then subtract  
that from every number, that means none of the  numbers are going to be big by definition because  
we've subtracted it from all of them. Now, the  problem is that's given us a different result,  
right? But if you think about it, let's expand  this sum. It's e to the power of x1, if we don't  
include our minus a, plus e to the power of  x2, plus e to the power of x3 and so forth.
Okay. Now, we just  
subtracted a from our exponents, which has meant  we're now wrong. But I've got good news. I've got  
good news and bad news. The bad news is that  you've got more high school math to remember,  
which is exponent rules. So x to the a  plus b equals x to the a times x to the b.  
And similarly, x to the a minus b  equals x to the a divided by x to the b.
And to convince yourself that's  true, consider, for example,   2 to the power of 2 plus 3. What is that? Well,  you've got 2 to the power of 2 is just 2 times 2.  
And 2 to the power of 2 plus 3, well,  it's 2 times 2 times, it is 2 to the  
power of 5. So you've got 2 to the power of  2, you've got two of them here, and you've   got another three of them here. So we're just  adding up the number to get the total index.
So we can take advantage of this here and say,  like, oh, well, this is equal to e to the x1 over  
e to the a plus e to the x2  over e to the a plus e to the x3  
over e to the a. And this is a common  denominator, so we can put all that together.  
e to the a. And why did we do all that? Because  if we now multiply that all by e to the a,  
these would cancel out and we get  the thing we originally wanted.   So that means we simply have to multiply this  
by that, and this gives us exactly  the same thing as we had before.  
But with, critically, this is no longer ever  going to be a giant number. So this might seem  
a bit weird. We're doing extra calculations. It's  not a simplification, it's a complexification.  
But it's one that's going to make it  easier for our floating point unit. So that's our trick, is rather than doing log  of this sum, what we actually do is log of  
e to the a times the sum of e to the x minus  a. And since we've got log of a product, that's  
just the sum of the logs, and log of e to the a  is just a. So it's a plus that. So this here is  
called the log sum exp trick.
Oops, people pointing out that  I've made a mistake. Thank you.  
That, of course, should have  been inside the log. You can't  
just go sticking it on the outside like a  crazy person. That's what I meant to say.
Okay, so here is the log sum exp trick. Oh,  I caught it m instead of a, which is a bit   silly. I should have caught it a. But anyway,  so we find the maximum on the last dimension,  
and then here is the m plus that exact thing.
Okay, so that's just another way of doing  that. Okay, so that's the logsumexp().
So now we can rewrite log_softmax() as x  minus logsumexp(), and we're not going to  
use our version because PyTorch already  has one, so we'll just use PyTorch's.  
And if we check, here we go. Here's our results.  
And so then as we've discussed, the cross  entropy loss is the sum of the outputs times  
the log probabilities. And as we discussed, our  outputs are one-hot encoded, or actually they're  
just the integers, better still. So what we can do  is we can, I guess I should make that more clear.  
Actually, they're just the integer indices.
So we can simply rewrite that  as negative log of the target.  
So that's what we have in our Excel. And so  how do we do that in PyTorch? So this is quite  
interesting. There's a lot of cool things you  can do with array indexing in PyTorch and NumPy,  
so basically they use the same  approaches. Let's take a look. Here is the first three actual values  in y_train. They're 5, 0, and 4.  
Now, what we want to do is we want to find  in our softmax predictions, we want to get  
5, the fifth prediction in the zeroth row,  
the 0 prediction in the first row, and  the 4 prediction in the index two row.
So these are the numbers that we want. This  is going to be what we add up for the first   two rows of our loss function. So how do we  do that in all in one go? Well, here's a cool  
trick. See here I've got 0, 1, 2. If we index  using a two lists, we can put here 0, 1, 2,  
and for the second list we can put y_train  column three: 5, 0, 4, and this is actually   going to return 0 comma 0, 1 comma… sorry, it's  going to be 0 comma 5, 1 comma 0, and 2 comma 4,  
which is, as you see, exactly the same thing. So therefore, this is actually giving us  what we need for the cross entropy loss.  
So if we take range of our target's  first dimension, or zero index dimension,  
which is all this is, and the target, and  then take the negative of that dot mean,  
that gives us our cross entropy loss,  which is pretty neat, in my opinion.
All right, so PyTorch calls this negative log  likelihood loss, but that's all it is. And so  
if we take the negative log likelihood  and we pass that to the log soft max,  
then we get the loss. And this  particular combination in PyTorch  
is called F.cross_entropy(). So just check, yep,  F.cross_entropy() gives us exactly the same thing.  
So that's cool. So we have now reimplemented the cross entropy  loss. And there's a lot of confusing things going  
on there, a lot. And so this is one of those  places where you should pause the video and go  
back and look at each step and think not just  like what is it doing, but why is it doing it,  
and also try typing in lots of different values  yourself to see if you can see what's going on,  
and then put this aside and test  yourself by reimplementing log_softmax()  
and nll_loss() and cross_entropy() yourself  and compare them to PyTorch's values.
And so that's a piece of  homework for you for this week.  
So now that we've got that, we can actually  create a training loop. So let's set our   loss function to be cross entropy. Let's  create a batch size of 64. And so here's  
our first mini batch. Okay, so xb is the x  mini batch. It's going to be from 0 up to 64  
from our training set. So we can now calculate  our predictions. So that's 64 by 10. So for  
each of the 64 images in the mini batch, we  have 10 probabilities, one for each digit.
And our Y is just, let's print those out.  
So there's our first 64 target values. So these  are the actual digits. And so our loss function,  
so we're going to start with a bad loss  because it's entirely random at this point.  
Okay, so for each of the predictions we made,  
so those are our predictions. And so  remember those predictions are a 64 by 10.  
What did we predict? So for each one of  these 64 rows, we have to go in and see  
where is the highest number. So if we go  through here, we can go through each one.  
Here's a, it's a 0.1. Okay, it looks like this  is the highest number. So it's 0, 1, 2, 3. So  
it's the highest number is this one. So you've  got to find the index of the highest number.   The function to find the index of the highest  number is called argmax. And yep, here it is, 3.  
And I guess we could have also written this  probably as preds.argmax(). Normally you can  
do them either way. I actually prefer normally  to do it this way. Yep, there's the same thing.
Okay, and the reason we want this is because we  want to be able to calculate accuracy. We don't   need it for the actual neural net, but we just  like to be able to see how we're going because  
it's like it's a metric. It's something that we  use for understanding. So we take the argmax,  
we compare it to the actual. So that's going  to give us a bunch of bools. If you turn those  
into floats, there'll be ones and zeros and  the mean of those floats is the accuracy. So our current accuracy, not surprisingly, is  around 10%. It's 9% because it's random. That's  
what you would expect. So let's train our first  neural net. So we'll set a learning rate. We'll  
do a few epochs. So we're going to go through  each epoch and we're going to go through from  
0 up to n. That's the 50,000 training rows.  And skipping by 64, the batch size, each time.  
And so we're going to create a slice that  starts at i, so starting at 0, and goes up to  
64 unless we've gone past the end, in which  case we'll just go up to n. And so then we  
will slice into our training set for the x  and for the y to get our x and y batches.  
We will then calculate our predictions,  our loss function, and do our backward.
So the way I did this originally was I had all  of these in… oopsie-daisy, in separate cells  
and I just typed in, you know, i equals zero  and then went through one cell at a time,  
calculating each one until they all worked.  And so then I can put them in a loop.
Okay, so once we've got  done backward, we can then,  
with torch.no_grad(), go through each layer  and if that's a layer that has weights,  
we'll update them to the existing weights  minus the gradients times the learning rate.  
And then zero out, so the weights and biases for  the gradients, the gradients of the weights and  
biases. This underscore means do it in place.  So that sets this to zero. So if I run that,  
oops, got to run all of  them. I guess I skipped cell.
There we go. It's finished.  
So you can see that our accuracy on  the training set, it's a bit unfair,  
but it's only three epochs, is nearly  97%. So we now have a digit recognizer.  
It trains pretty quickly and is not terrible  at all. So that's a pretty good starting point.
All right, so what we're going to do next time is  we're going to refactor this training loop to make  
it dramatically, dramatically, dramatically  simpler, step by step until, eventually,  
we will get it down to, so we'll get it down  to something much, much shorter. And then  
we're going to add a validation set to it and  a multi-processing data loader, and then, yeah,  
we'll be in a pretty good position, I think,  to start training some more interesting models.
All right. Hopefully you found that useful  and learned some interesting things. And so  
what I'd really like you to do is, at this  point, now that you've kind of like got all  
these key basic pieces in place, is to really  try to recreate them without peaking as much  
as possible. So, you know, recreate your matrix  multiply, recreate those forward and backward  
passes, recreate something that steps through  layers, and even see if you can like recreate  
the idea of the dot forward and the dot backward.  Make sure it's all in your head really clearly so  
that you fully understand what's going on. You  know, at the very least, if you don't have time  
for that, because that's a big job, you could pick  out a smaller part of that, the piece that you're   more interested in, or you could just go through  and look really closely at these notebooks. So if  
you go to kernel, restart and clear output, it'll  delete all the outputs and like try to think like  
what are the shapes of things? Can you guess  what they are? Can you check them? And so forth.
Okay. Thanks, everybody. Hope you have a  great week and I will see you next time. Bye.

---

# 14

﻿Okay, hi everybody and welcome to Lesson 14.  The numbers are getting up pretty high now, huh?  
We had a lesson last time talking about  calculus and how we implement the chain rule  
in neural network training in an efficient way  called backpropagation. I just wanted to point  
out that one excellent student, Kaushik Sinha,  has produced a very nice explanation of the  
Review of code and math from Lesson 13
code that we looked at last time and I've linked  to it. So it's got the math and then the code.  
The code's slightly different to what I had, but  it's basically the same thing, some minor changes.   And it might be helpful to kind of link between  the math and the code to see what's going on.  
So you'll find that in the Lesson 13  resources. But I thought I'd just quickly  
try to explain it as well. So maybe I could try to  copy this and just explain what's going on here.  
With this code. So the basic idea is that we  have a neural network that is calculating,  
well, a neural network and a loss function that  together they calculate a loss. So let's imagine,  
let’s just call the loss  function, we'll call it L.  
And the loss function is being applied to the  output of the neural network. So the neural   network function we'll call n. And that takes two  things, a bunch of weights and a bunch of inputs.  
The loss function also requires the targets, but  I'm just going to ignore that for now because  
it's not really part of what we actually care  about. And what we're interested in knowing is  
if we want to be able to update the weights,  let's say this is just a single layer things,   keep it simple. If we want to be able to update  the weights, we need to know how does the  
loss change if we change the weights, if we  change one weight at a time, if you like.  
So how would we calculate that? Well, what we  could do is we could rewrite our loss function  
by saying, well, let's call capital N the result  of the neural network applied to the weights  
and the inputs. And that way we can now  rewrite the loss function to say L equals,  
big L equals, little l, the loss function  applied to the output of the neural network.  
And so maybe you can see where this is going.  We can now say, okay, the derivative of the  
loss with respect to the weights is going  to be equal to the derivative of the loss  
with respect to the outputs  of that neural network layer  
times, this is the chain rule, the derivative  of the outputs of that neural network layer.  
I'm going to get my notation consistent since  these are not scalar with respect to the weights.  
So you can see we can get rid of those and we  end up with the change in loss with respect  
to the weights. And so we can just say this is  a chain rule. This is what the chain rule is.  
So the change in the loss with respect  to the output of the neural network,  
well, we did the forward pass here and  then we took here, this here is where  
we calculated the derivative of the loss with  respect to the output of the neural network,  
which came out from here and ended up in diff. So  there it is. So out.g contains this derivative.  
So then to calculate, let's actually do one  more. We could also say the change in the  
loss with respect to the inputs, we can do  the same thing with the chain rule times…  
And so this time we have the  inputs. So here you can see that is  
this line of code. So that is the change  in the loss with respect to the inputs.  
That's what input.g means. And it's equal to the  change in the loss with respect to the output.  
So that's what out.g means. Times… It's  actually matrix times, because we're doing  
matrix calculus, times this derivative, and  since this is a linear layer we were looking at,  
this derivative is simply the weights themselves.  And then we have exactly the same thing  
for w.g, which is the change in the loss,  the derivative of the loss with respect to  
the weights. And so again, you've got the same  thing. You've got your out.g, and remember we  
actually showed how we can simplify this into  also a matrix product with a transpose as well.  
So that's how what's happening in  our code is mapping to the math.  
So hopefully that's useful, but as I say, do check  out this really nice resource, which has a lot  
more detail if you're interested in digging  deeper. The other thing I'd say is if you,  
some people have mentioned that they actually  didn't study this at high school, which is fine.  
We've provided resources on the forum for  recommending how to learn the basics of  
derivatives and the chain rule. And so in  particular, I would recommend 3Blue1Brown's  
essence of calculus series and also Khan  Academy. It's not particularly difficult  
to learn. It'll only take you a few hours and  then you can, this will make a lot more sense.  
Or if you did it at high school,  but you've forgotten it, same deal.   So don't worry if you found this difficult because  you had forgotten the, or had never learned  
the basic derivative and chain rule stuff. That's  something that you can pick up now and I would  
recommend doing so. Okay. So what we then did last  time, which is actually pretty exciting, is we got  
f-Strings
to a point where we had successfully created  a training loop, which did these four steps.  
So and the nice thing is that every  single thing here is something that   we have implemented from scratch. Now, we didn't  always use our implemented from scratch versions.  
There's no particular reason to, when we've  re-implemented something that already exists,   let's use the version that exists. But every  single thing here, well, I guess not argmax,  
but that's trivially easy to implement. Every  single thing here, we have implemented ourselves  
and we successfully trained an MNIST model to  96% accurately recognize handwritten digits.  
So I think that's super neat. It's, this  is, I mean, this is not a great metric.  
It's only looking at the training set, in  particular it's only looking at one batch   of the training set. Since last time, I've just  refactored a little bit. I've pulled out this  
report function, which is now just running at  the end of each epoch. And it's just printing  
out the loss and the accuracy. Just something  I wanted to mention here is hopefully you've  
seen f-strings before. They're a really helpful  part of Python that lets you pop a variable or  
an expression inside curly braces in a string and  it'll evaluate it. You might not have seen this  
colon thing. This is called a format specifier.  And with a format specifier, you can change how  
things are printed in an f-string. So this is  how I'm printing it to do decimal places. This  
says a two decimal places floating point number  called loss printed out here, followed by a comma.  
So I'm not going to show you how to use those  other than to say, yeah, Python f-strings and  
format specifiers are really helpful. And so if  you haven't used them before, do go look them up,   a tutorial of the documentation, because they're  definitely something that you'll probably find  
useful to know about. Okay. So let's  just rerun all those lines of code.  
Re-running the Notebook - Run All Above
If you're wondering how I just reran  all the cells above where I was,  
there's a cell here. There's Run All Above.  And it's so helpful that I always make sure  
there's a keyboard shortcut for that. So you  can see here, I've added a keyboard shortcut  
QA. So if I type QA, it runs all cells above. If  I type QB, it runs all cells below. And so yeah,  
stuff that you do a lot, make sure you've got  keyboard shortcuts for them. You don't want   to be fiddling around, moving around your mouse  everywhere. You want it to be as easy as thinking.  
So this is really exciting. We've successfully  built and trained a neural network model from   scratch and it works okay. It's a bit clunky.  There's a lot of code. There's features we're  
missing. So let's start refactoring it. And  so refactoring is all about making it so we  
Starting code refactoring: torch.nn
have to write less code to do the same work.  And so we're now going to, I'm going to show  
you something that's part of PyTorch and  then I'm going to show you how to build it.   And then you'll see why this is really useful. So  PyTorch has a sub module called nn, torch.nn. And  
in there, there's something called the Module  class. Now we don't normally use it this way,   but I just want to show you how it works. We  can create an instance of it in the usual way  
where we create instances of classes, and then we  can assign things to, attributes of that module.  
So for example, let's assign a linear  layer to it. And if we now print out  
that, you'll see it says, oh, this is a  module containing something called foo,   which is a linear layer. But here's something  quite tricky. This module, we can say,  
show me all of the named children  of that module. And it says, oh,   there's one called foo and it's a linear layer.  And we can say, oh, show me all of the parameters  
of this module. And it says, oh, okay,  sure. There's two of them. There's this  
four by three tensor, that's the weights. And  there's this four long vector, that's the biases.  
And so somehow just by creating this module  and assigning this to it, it's automatically  
tracked what's in this module and what are its  parameters. That's pretty neat. So we're going to  
see both how and why it does that. I'm just going  to point out, by the way, why did I add list here?  
Generator Object
If I just said m1.named_children(), it just  prints out generator object, which is not very  
helpful. And that's because this is a kind of  iterator called a generator. And it's something  
which is going to only produce the contents  of this when I actually do something with it,  
such as list them out. So just popping a list  around a generator is one way to run the generator  
and get its output. So that's a little trick  when you want to look inside a generator.  
Class MLP: Inheriting from nn.Module
Okay. So now, as I said, we don't normally use  it this way. What we normally do is we create  
our own class. So for example, we'll create  our own multi-layer perceptron and we inherit   it. We inherit from nn.Module. And so then in  dunder init, this is the thing that constructs  
an object of the class. This is the special magic  method that does that. We'll say, okay, well,  
how many inputs are there to this multi-layer  perceptron? How many hidden activations and how  
many output activations are there? So it'd just  be one hidden layer. And then here we can do just   like we did up here, where we assigned things as  attributes, we can do that in this constructor.  
So we'll create an l1 attribute, which is a  linear layer from number in to number hidden.  
l2 is a linear layer from number hidden to  number out, and we'll also create a ReLU. And so,  
when we call that module, we  can take the input that we get  
and run the linear layer and then run the ReLU and  then run the l2. And so I can create one of these,  
as you see, and I can have a look and see like,  oh, here's the attribute l1. And there it is,  
like I had, and I can say, print out the model and  the model knows all the stuff that's in it. And  
I can go through each of the named children and  print out the name and the layer. Now, of course,  
if you remember, although you can use dunder call,  we actually showed how we can refactor things  
using forward such that it would automatically  kind of do the things necessary to make all the  
automatic gradient stuff work correctly. And  so in practice, we're actually not going to do  
dunder call, we would do forward. So this is an  example of creating a custom PyTorch module. And  
the key thing to recognize is that it knows  what are all the attributes you added to it.  
And it also knows what are all the parameters. So  if I go through the parameters and print out their  
shapes, you can see I've got my linear layers  weights, first linear layer, sorry, second linear  
layer, my… oh no: first linear layers weights, my  first linear layers biases, second linear layers  
weights, second linear layers biases. And this 50  is because we set nh, the number of hidden, to 50.  
So why is that interesting? Well, because  now I don't have to write all this anymore  
going through layers and having to make  sure that they've all been put into a list.  
We've just been able to add them as  attributes and they're automatically   going to appear as parameters. So we  can just say, go through each parameter  
and update it based on the gradient  and the learning rate. And furthermore,  
you can actually just go model.zero_grad()  and it'll zero out all of the gradients.  
Checking the more flexible refactored MLP
So that's really made our code quite a lot nicer  and quite a lot more flexible, which is cool.  
So let's check that this still works.  
There we go. So just to clarify, if I called  report() on this before I ran it, as you would  
expect, the accuracy is about 8%, well, about  10%, a bit less, and the loss is pretty high.  
And so after I run this fit(), this model,  the accuracy goes up and the loss goes down.  
So basically it's all of this  exactly the same as before.   The only thing I've changed are these two lines  of code. So that's a really useful refactoring.  
So how on earth did this happen? How did it know  what the parameters and layers are automatically?  
Creating our own nn.Module
It used a trick called dunder setattr, and  we're going to create our own nn.Module now.  
So if there was no such thing as  nn.Module, here's how we'd build it.   And so let's actually build it and also add some  things to it. So in dunder init, we would have to  
create a dictionary for our named children. This  is going to contain a list, a dictionary of all  
of the layers. Okay. So just like before,  we'll create a couple of linear layers,   right? And then what we're going to do is we're  going to define this special magic thing that  
Python has called dunder setattr. And this is  called automatically by Python, if you have it,   every time you set an attribute such as here or  here. And it's going to be passed the name of the  
attribute, the key, and the value is the actual  thing on the right hand side of the equal sign.  
Now, generally speaking, things that start with an  underscore we use for private stuff. So we check  
that it doesn't start with an underscore. And if  it doesn't start with an underscore, setattr will  
put this value into the modules dictionary  with this key and then call Python’s…  
the normal Python’s setattr to make sure it  just actually does the attribute setting.  
So super() is how you call whatever  is in the super class, the base class.  
So another useful thing to know about is how does  it do this nifty thing where you can just type the  
name and it kind of lists out all this information  about it. That's a special thing called dunder  
repr. So here dunder repr will just have it return  a stringified version of the modules dictionary.  
And then here we've got parameters(). How did  parameters work? So how did this thing work? Well,  
we can go through each of those modules,  go through each value. So the values of the  
modules is all the actual layers and then go  through each of the parameters in each module  
and yield p. So that's going to create an  iterator, if you remember when we looked  
at iterators for all the parameters. So let's  try it. So we can create one of these modules  
and if we just like before loop  through its parameters, there they are.  
Now I'll just mention something that's optional,  kind of like advanced Python that a lot of people  
don't know about, which is there's no need to  loop through a list or a generator or I guess  
say loop through an iterator and yield. There's  actually a shortcut, which is you can just say:  
yield from and then give it the iterator.  And so with that, we can get this all down  
to one line of code and it'll do exactly  the same thing. So that's basically saying  
yield one at a time, everything in here, that's  what yield from does. So there's a cool little  
advanced Python thing, totally optional, but if  you're interested, I think it can be kind of neat.  
So we've now learned how to create our own  implementation of nn.Module and therefore  
we are now allowed to use PyTorch's  nn.Module. So that's good news. So  
Using PyTorch’s nn.Module
how would we do using the PyTorch nn.Module, how  would we create the model that we started with,  
which is where we had this self.layers? Because we  want to somehow register all of these all at once.  
That's not going to happen  based on the code we just wrote.  
So to do that, let's have a look. We can, so  let's make a list of the layers we want. And so  
we'll create again a subclass of nn.Module. Make  sure you call the super() classes in it first,  
and we'll just store the list of layers and  then to tell PyTorch about all those layers,  
we basically have to loop through them and call   add_module() and say what the name of  the module is and what the module is.  
And again, probably should have used  forward here in the first place.  
And you can see this has now done exactly the same  thing. Okay. So if you've used a sequential model  
before, you'll see, or you can see that we're  on the path to creating a sequential model.  
Okay. So Ganesh has asked an interesting  question, which is what on earth is super  
calling because we actually, in fact, we  don't even need the parentheses here. We   actually don't have a base class. That's  because if you don't put any parentheses  
or if you put empty parentheses, it's actually  a shortcut for writing that. And so Python has  
stuff in object, which does, you know,  all the normal objecty things like  
storing your attributes so that you can get them  back later. So that's what's happening there.  
Okay. So this is a little bit awkward is to  have to store the list and then enumerate and  
call add_module(). So now that we've implemented  that from scratch, we can use PyTorch's version,  
Using PyTorch’s nn.ModuleList
which is they've just got something called  ModuleList that just does that for you.   Okay. So if you use ModuleList and pass it  a list of layers, it will just go ahead and  
register them all those modules for you. So  here's something called SequentialModel. This   is just like nn.Sequential now. So if I create it  passing in the layers, there you go. You can see  
there's my model containing my  module list with my layers. And so,  
I don't know why I never used  forward for these things.   It's silly. I guess it doesn't matter  terribly in this stage, but anywho. Okay. So,  
call fit() and there we go. Okay. So in forward  here, I just go through each layer and I set the  
result of that equal to calling that layer on the  previous result and then pass and return it at the  
end. Now there's a little, another way of doing  this, which I think is kind of fun. It's not,  
like, shorter or anything at this stage. I just  wanted to show an example of something that you   see quite a lot in machine learning code, which  is the use of reduce(). This implementation here  
reduce()
is exactly the same as this thing here.  
So let me explain how it works. What reduce  does. Reduce is a very common kind of, like,  
fundamental computer science concept: reductions.  This is something that does a reduction and what  
a reduction is, is it's something that says, start  with, the third parameter, some initial value. So  
we're going to start with x, the thing with being  passed and then loop through a sequence. So look  
through each of our layers and then for each  layer call some function. Here is our function.  
And the function is going to get passed, first  time around, it’ll be passed the initial value  
and the first thing in your  list. So your first layer and x.   So it's just going to call the layer function on  x. The second time around it takes the output of  
that and passes it in as the first parameter and  passes in the second layer. So then the second  
time this goes through, it's going to be calling  the second layer on the result of the first layer  
and so forth. And that's what a reduction  is. And so when you might see reduce(),   you'll certainly see it talked about quite a lot  in papers and books, and you might sometimes also  
see it in code. It's a very general concept. And  so here's how you can implement a sequential model  
using reduce(). So there's no explicit loop there,  although the loop is still happening internally.  
PyThorch’s nn.Sequential
So now that we've reimplemented sequential, we  can just go ahead and use PyTorch's version.   So there's nn.Sequential. We can pass in our  layers and we can fit, not surprisingly. We  
can see the model. So yeah, it looks very  similar to the one we built ourselves.  
All right. So this thing of  looping through parameters  
and updating our parameters based on gradients  and a learning rate, and then zeroing them is  
very common. So common that there is something  that does that all for us. And that's called an  
Optimizer
optimizer. It's the stuff in optim. So let's  create our own optimizer. And as you can see,  
it's just going to do the two things we just saw.  It's going to go through each of the parameters  
and update them using the gradient and the  learning rate. And there's also zero grad,   which will go through each parameter  and set their gradients to zero.  
If you use .data, it's just a way of avoiding  having to say torch.no_grad, basically.  
So in optimizer, we're going to pass it  the parameters that we want to optimize,   and we're going to pass it the learning rate.  And we're just going to store them away. And  
since the parameters might be a generator,  we'll call list() to turn them into a list.  
So we're going to create our optimizer, pass  it in the model.parameters(), which have been  
automatically constructed for us by nn.Module.  And so here's our new loop. Now we don't have  
to do any of the stuff manually. We can just  say opt.step(). So that's going to call this.  
And opt.zero_grad(). And  that's going to call this.  
There it is. So we've now built our own SGD  optimizer from scratch. So I think this is  
really interesting, right? These things which  seem like they must be big and complicated,  
once we have this nice structure in place,  an SGD optimizer doesn't take much code at  
all. And so it's all very transparent,  simple, clear. If you're having trouble  
using complex library code that you've found  elsewhere, this can be a really good approach,  
is to actually just go all the way back, remove  as many of these abstractions as you can and  
run everything by hand to see exactly what's  going on. It can be really freeing to see that  
you can do all this. Anyway, since PyTorch  has this for us in torch.optim, it's got an  
PyTorch’ optim and get_model()
optim.SGD(). And just like our version, you pass  in the parameters and you pass in the learning  
rate. So you really see it is just the same.  So let's define something called get_model().  
That's going to return the model, the sequential  model and the optimizer for it. So if we go model,  
opt equals get_model(), and then we can call  the loss function to see where it's starting.  
Dataset
And so then we can write our training loop  again. Go through each epoch, go through each  
starting point for our batches, grab the slice,  slice into our X and Y in the training set,  
calculate our predictions, calculate our loss,  do the backward pass, do the optimizer step,  
do the zero gradient and print out how you're  going at the end of each one. And there we go.  
All right. So let's keep making this  simpler. There's still too much code.   So one thing we could do is we could replace  these lines of code with one line of code by using  
something we'll call the Dataset class. So the  Dataset class is just something that we're going   to pass in our independent and dependent variable.  We'll store them away as self.x and self.y.  
We'll have something. So if you define dunder len,  then that's the thing that allows the len function  
to work. So the length of the Dataset will just  be the length of the independent variables.   And then dunder getitem is the thing that  will be called automatically anytime you  
use square brackets in Python. So that just  is going to call this function passing in  
the indices that you want. So when  we grab some items from our Dataset,   we're going to return a tuple of the x values and  the y values. So then we'll be able to do this.  
So let's create a Dataset using  this tiny little tree line class.  
It's going to be a Dataset containing the  x and y training, and then create another   Dataset containing the x and y valid. And those  two datasets we'll call train_ds and valid_ds.  
So let's check the length of those datasets should  be the same as the length of the x’s and they are.  
And so now we can do exactly what  we hoped we could do. We can say xb   comma yb equals train_ds and pass in some slice.  
So that's going to give us back our… Check the  shapes are correct. It should be 5 by 28 by 28,  
5 by 28 times 28 and the y should just be  five. And so here they are the x’s and the y’s.  
So that's nice. We've created a Dataset from  scratch. And again, it's not complicated  
at all. And if you look at the actual PyTorch  source code, this is basically all Dataset do.  
So let's try it. We call get_model(). And so now  we've replaced our dataset line with this one and  
per usual, it still runs. And so this is what  I do when I'm writing code is I try to, like,  
always make sure that my starting code works as  I refactor. And so you can see all the steps.  
And so somebody reading my code can then see  exactly like, why am I building everything   I'm building? How does it all fit in? See that  it still works. And I can also keep it clear  
in my own head. So I think this is a really  nice way of implementing libraries as well.  
DataLoader
All right. So now we're going to  replace these two lines of code  
with this one line of code. So we're going  to create something called a DataLoader and   a DataLoader is something that's just going to  do this. Okay. So we need to create an iterator.  
So an iterator is a class that has a dunder  iter method. When you say “for in” in Python,  
behind the scenes, it's actually calling dunder  iter to get a special object, which it can then  
loop through using yield. So it's basically  getting this thing that you can iterate   through using yield. So a DataLoader is something  that's going to have a Dataset and a batch size,  
because we're going to go through the  batches and grab one batch at a time.  
So we have to store away the Dataset and the batch  size. And so when you, when we call the for loop,  
it's going to call dunder iter. We're going to  want to do exactly what we saw before, go through   the range, just like we did before, and then  yield that bit of the data set. And that's all.  
So that's a DataLoader. So we can now create  a train DataLoader and a valid DataLoader from  
our train Dataset and valid Dataset. And so  now we can, if you remember the way you can  
create one thing out of an iterator, so you don't  need to use a for loop, you can just say iter,  
and that will also call dunder iter. Next, we'll  just grab one value from it. So here we will run  
this and you can see we've now just confirmed  we’ve xb is a 50 by 784 and yb, there it is.  
And then we can check what it looks like. So  let's grab the first element of our X batch,   make it 28 by 28. And there it is. So now that  we've got a DataLoader, again, we can grab our  
model and we can simplify our fit function to just  go for xb, yb in train_dl. So this is getting nice  
and small, don't you think? And it still works  the same way. Okay. So this is really cool.  
And now that it's nice and concise,  we can start adding features to it.   So one feature I think we should add is that  our training set, each time we go through it,  
Random sampling, batch size, collation
it should be in a different order. It should  be randomized, the order. So instead of always  
just going through these indexes in order, we  want some way to say, go use random indexes.  
So the way we can do that is create a class  called Sampler. And what sampler is going to do,  
I'll show you, is if we create a sampler  without shuffle, without randomizing it,  
it's going to simply return all  the numbers from zero up to n  
in order and it'll be an iterator. See, this  is dunder iter. But if I do want it shuffled,  
then it will randomly shuffle them. So here you  can see I've created a sampler without shuffle.  
So if I then make an iterator from that and print  a few things from the iterator, you can see it's  
just printing out the indexes it's going to  want. Or I can do exactly the same thing as  
we learned earlier in the course using islice. We  can grab the first five. So here's the first five  
things from a sampler when it's not shuffled.  So as you can see, these are just indexes.  
So we could add shuffle equals true. And now  that's going to call random.shuffle(), which just  
randomly permuts them. And now if I do the same  thing, I've got random indexes of my source data.  
So why is that useful? Well, what we could now  do is create something called a BatchSampler.  
And what the BatchSampler is going to do is it's  going to basically do this islice thing for us. So  
we're going to say, okay, pass in a sampler.  So that's something that generates indices   and pass in a batch size. And remember, we've  looked at chunking before. It's going to chunk  
that iterator by that batch size.  
And so if I now say, all right, please  take our sampler and create batches of 4.  
As you can see here, it's creating batches  of four indices at a time. So rather than  
just looping through them in order, I  can now loop through this BatchSampler.  
So we're going to change our data loader  so that now it's going to take some  
BatchSampler. And it's going to loop through the  BatchSampler. That's going to give us indices.  
And then we're going to get that Dataset item  from that batch for everything in that batch.  
So that's going to give us a list. And then we  have to stack all of the x’s and all of the y’s  
together into tensors. So I've created  something here called collate function.  
And we're going to default that to this little  function here, which is going to grab our  
batch, pull out the x’s and y’s separately,  and then stack them up into tensors.  
So this is called our collate function.  Okay. So if you put all that together,  
we can create a training sampler, which is a batch  sampler over the training set with shuffle true.  
A validation sampler will be a batch sampler  over the validation set with shuffle false.  
And so then we can pass that  into this DataLoader class,  
the training data set and the training sampler  and the collate function, which we don't really  
need because it's, we're just using the default  one. So I guess we can just get rid of that.  
And so now here we go. We can do  exactly the same thing as before xb,   yb, next, iter. And this time we use the valid  DataLoader, check the shapes. And this is how  
PyTorch's actual DataLoader works. This is the,  this is all the pieces they have. They have  
samplers, they have batch samplers, they have  a collation function and they have DataLoaders.  
So remember that what I want you  to be doing for your homework  
is experimenting with these carefully to see  exactly what each thing's taking in. Okay.  
What does collate do?
So Piotr is asking on the chat, what is this  collate thing doing? Okay. So collate function,  
it defaults to collate. What does it do? Well,  let's see, let's go through each of these  
steps. Okay. So we need, so we've got a batch  sampler, so let's do just the valid sampler.  
Okay. So the batch sampler,   here it is. So we're going to go through each  thing in the batch sampler. So let's just grab  
one thing from the batch sampler. Okay. So the  output of the batch sampler will be next. It's  
okay. So here's what the batch sampler  contains. All right. Just the first 50  
digits, not surprisingly, because this is our  validation sampler. If we did a training sampler,   that would be randomized. There they  are. Okay. And what we then do is we go  
self.dataset[i] for i in b.  So let's copy that. Copy,  
paste. And so rather than self.dataset[i],  we'll just say valid_ds[i].  
Oh, and it's not i and b it's i  and o that's what we called it.  
Oh, and we did it for training. Sorry. Training.  Okay. So what it's created here is a list of  
tuples of tensors, I think. Let's have a  look. So let's have a look. So let's say this,  
p —whatever. So p[0]. Okay is a tuple. It's got  the x and the y, independent variable. So that's  
not what we want. What we want is something that  we can loop through. We want to get batches. So  
what the collation model is going to do, sorry not  collation model, the collate function is going to  
do is it's going to take all of our x’s and all  of our y’s and collate them into two tensors,  
one tensor of x’s and one tensor of y’s. So the  way it does that is it first of all calls zip().  
So zip is a very, very commonly used Python  function. It's got nothing to do with the  
compression program zip, but instead what it does  is it effectively allows us to transpose things  
so that now, as you can see, we've got all of the  second elements or index 1 elements all together  
and all of the index 0 elements together. And  so then we can stack those all up together  
and that gives us our y’s for our batch. So that's  what collate does. So the collate function is used  
an awful lot in PyTorch, increasingly nowadays  where Hugging Face stuff uses it a lot. And so  
we'll be using it a lot as well. And basically  it's a thing that allows us to customize how the  
data that we get back from our Dataset, once  it's been kind of generating a list of things  
from the Dataset, how do we put it together  into a bunch of things that our model can  
take as inputs? Because that's really what we want  here. So that's what the collation function does.  
Oh, this is the wrong way around.  
Like so. This is something that I do so often that  fastcore has a quick little shortcut for it, just  
fastcore’s store_attr()
called store_attr, store attributes. And so if you  just put that in your dunder init, then you just  
need one line of code and it does exactly the same  thing. So there's a little shortcut as you see.  
And so you'll see that quite a bit. All  right. Let's have a seven minute break and  
see you back here very soon. And we're going  to look at a multi-processing DataLoader,  
and then we'll have nearly finished  this notebook. All right. See you soon.  
All right. Let's keep going. So we've seen how  to create a DataLoader and sampling from it.  
Multiprocessing DataLoader
The PyTorch DataLoader works exactly like this,  but it uses a lot more code because it implements  
multi-processing. And so multi-processing means  that the actual, this thing here, that code,  
can be run in multiple processes. They can be run  in parallel for multiple items. So this code, for  
example, might be opening up a JPEG, rotating it,  flipping it, et cetera. Right? So because remember  
this is just calling the dunder getitem for a  Dataset. So that could be doing a lot of work for  
each item and we're doing it for every item in the  batch. So we'd love to do those all in parallel.   So I'll show you a very quick and dirty  way that basically does the job. So  
Python has a multi-processing library. It doesn't  work particularly well with PyTorch tensors. So  
PyTorch has created an exact re-implementation of  it. So it's identical API wise, but it does work  
well with tensors. So this is basically what  has grabbed the multi-processing. So this is   not quite cheating because multi-processing isn't  the standard library and this is API equivalent.  
So I'm going to say, we're allowed to do that.  So as we've discussed, you know, when we call  
square brackets on a class, it's actually  identical to calling the dunder getitem function  
on the object. So you can see here, if  we say, give me items 3, 6, 8, and 1,  
it's the same as calling dunder  getitem passing in 3, 6, 8, and 1.  
Now why does this matter? Well, I'll show you why.  It matters because we're going to be able to use  
map and I'll explain why we want to use map in  a moment. Map is a really important concept. You   might've heard of map-reduce. So we've already  talked about reductions and what those are. Maps  
are kind of the other key piece. Map is something  which takes a sequence and calls a function on  
every element of that sequence. So imagine we had  a couple of batches of indices, 3 and 6 and 8 and  
1. Then we're going to call dunder getitem  on each of those batches. So that's what map  
does. Map calls this function on every element  of the sequence. And so that's going to give us  
the same stuff, but now this same as this, but now  batched into two batches. Now why do we want to do  
that? Because multiprocessing has something called  Pool where you can tell it how many workers do you  
want to run, how many processes you want to run.  And it then has a map which works just like the  
normal Python map, but it runs this function  in parallel over the items from this iterator.  
So this is how we can create a multiprocessing  DataLoader. So here we're creating our DataLoader.  
And again, we don't actually need to pass in the  collate function because we're using the default   one. So if we say n_workers equals 2 and then  create that, if we say next, see how it's taking  
a moment and it took a moment because it was  firing off those two workers in the background.   So the first batch actually comes out more slowly.  But the reason that we would use a multiprocessing  
DataLoader is if this is doing a lot of work, we  want it to run in parallel. And even though the  
first item might come out a bit slower,  once those processes are fired up, it's   going to be faster to run. So this is a really  simplified multiprocessing DataLoader. Because  
this needs to be super, super efficient,  PyTorch has lots more code than this to make  
it much more efficient. But the idea is this,  and this is actually a perfectly good way of  
experimenting or building your own DataLoader  to make things work exactly how you want.  
PyTorch’s Multiprocessing DataLoader
So now that we've re-implemented all this from  PyTorch, let's just grab PyTorch’s. As you can  
see, they're exactly the same DataL oader. They  don't have one thing called sampler that you pass   shuffle to. They have two separate classes  called SequentialSampler and RandomSampler.  
I don't know why they do it that way. It's a  little bit more work to me, but same idea. And  
they've got BatchSampler. And so it's exactly the  same idea. The training sampler is a BatchSampler  
with a RandomSampler. The validation sampler  is a BatchSampler with a SequentialSampler.  
Pass them in batch sizes. And so we can now pass  those samplers to the DataLoader. This is now  
the PyTorch’s DataLoader. And just like ours, it  also takes a collate function. And it works. Cool.  
So that's, as you can see, it's doing exactly  the same stuff that ours is doing with exactly   the same API. And it's got some shortcuts, as I'm  sure you've noticed when you've used DataLoaders.  
So for example, calling batch sampler is going  to be very, very common. So you can actually just  
pass the batch size directly to a DataLoader, and  it will then auto-create the batch samplers for  
you. So you don't have to pass in BatchSampler at  all. Instead you can just say sampler, and it will  
automatically wrap that in the batch sampler  for you. So it does exactly the same thing.   And in fact, because it's so common to create  a RandomSampler or a SequentialSampler for a  
Dataset, you don't have to do that manually.  You can just pass in shuffle equals true or   shuffle equals false to the DataLoader. And  that does, again, exactly the same thing.  
There it is. Now something that is very  interesting is that, when you think about it,  
the batch sampler and the collation function  are things which are taking the result  
of the sampler, looping through them,  and then collating them together.  
But what we could do is, actually, because  
our Datasets know how to grab multiple  indices at once, we can actually just use  
the BatchSampler as a sampler. We don't  actually have to loop through them and  
collate them because they're basically instantly,  they come pre-collated. So this is a trick which  
actually Hugging Face stuff can use as well, and  we'll be seeing it again. So this is an important   thing to understand is how come we can pass a  BatchSampler to sampler and what's it doing? And  
so rather than trying to look through the  PyTorch code, I suggest going back to our   non-multi-processing pure Python code to see  exactly how that would work. Because it's a really  
nifty trick for things that you can grab multiple  things from at once and it can save a whole lot  
of time. It can make your code a lot faster. Okay.  So now that we've got all that nicely implemented,  
Validation set
we should now add a validation set. And there's  not really too much to talk about here. We'll   just take our fit function, and this is  exactly the same code that we had before.  
And then we're just going to add something  which goes through the validation set  
and gets the predictions and sums up the losses  and accuracies and from time to time prints out  
the loss and accuracy. And so get_dls(), we will  implement by using the PyTorch DataLoader now.  
And so now our whole process will be get_dls()  passing in the training and validation dataset.  
Notice that for our validation DataLoader, I'm  doubling the batch size because it doesn't have to   do back propagation. So it should use about half  as much memory so I can use a bigger batch size.  
Get our model and then call this  fit. And now it's printing out  
the loss and accuracy on the validation set.  So finally we actually know how we're doing,  
which is that we're getting 97% accuracy on the  validation set. And that's on the whole thing, not  
just on the last batch. So that's cool. We've now  implemented a proper, working, sensible training  
loop. It's still, you know, a bit more code  than I would like, but it's not bad. And every  
line of code in there and every line of code it's  calling is all stuff that we have built ourselves,  
re-implemented ourselves. So we know exactly  what's going on and that means it's going to be   much easier for us to create anything we can think  of. We don't have to rely on other people's code.  
So hopefully you're as excited about that as I  am. Cause it really opens up a whole world for us.  
Hugging Face Datasets, Fashion-MNIST
So one thing that we're going to want to be able  to do now that we've got a training loop is to  
grab data. And there's a really fantastic library  of datasets available on Hugging Face nowadays.  
And so let's look at how we use those datasets  now that we know how to bring things into data  
loaders and stuff so that now we can use  the entire world of Hugging Face datasets  
with our code. So we're going to,  so you need to pip install datasets.  
And once you've pip install datasets, you'll be  to say from datasets import, and you can import  
a few things. I just, these two things now,  load_dataset, load_dataset_builder. And we're   going to look at a dataset called Fashion-MNIST.  And so the way things tend to work with Hugging  
Face is there's something called the Hugging  Face hub, which has models and it has datasets  
amongst other things. And generally you'll give  them a name and you can then say, in this case,  
load a dataset builder for Fashion-MNIST. Now a  dataset builder is just basically something which  
has some metadata about this dataset. So the  dataset builder has a .info and the .info has a  
.description. And here's a description of this.  And as you can see, again, we've got 28 by 28  
grayscale. So it's going to be very familiar  to us because it's just like MNIST. And again,   we've got 10 categories. And again, we've got  60,000 training examples. And again, we've got  
10,000 test examples. So this is cool. So as it  says, it's a direct drop-in replacement for MNIST.  
And so the dataset builder also will tell  us what's in this dataset. And so Hugging  
Face stuff generally uses dictionaries rather  than tuples. So there's going to be an image   of type Image, and there's going to be a label of  type ClassLabel There's 10 classes and these are  
the names of the classes. So it's quite nice that  in Hugging Face datasets, you know, we can kind  
of get this information directly. It also tells us  if there are some recommended training test bits,  
we can find out those as well. So this is the size  of the training split and the number of examples.  
So now that we're ready to start playing with  it, we can load the dataset. Okay, so this is the  
difference between load_dataset_builder() versus  load_dataset(). So this will actually download it,   cache it, and here it is. And it creates a dataset  dictionary. So a dataset dictionary, if you've  
used fast.ai, is basically just like what we call  the datasets class. They call the DatasetDict  
class. So it's a dictionary that contains in  this case, a train and a test item, and those are  
datasets. These datasets are very much like the  datasets that we created in the previous notebook.  
So we can now grab the training and test items  from that dictionary and just pop them into  
variables. And so we can now have a look at the  0 index thing in training. And just like we were  
promised, it contains an image and a label.  So as you can see, we're not getting tuples  
anymore. We're getting dictionaries containing  the x and the y, in this case, image and label.  
So I'm going to get pretty bored writing image and  label in strings all the time. So I'm just going  
to store them as x and y. So x is going to be the  string ‘image’ and y will be the string ‘label’.  
I guess the other way I could have done that   would have been to say x comma y equals  that. That would probably be a bit neater  
because it's coming straight from the  features. And if you iterate into a dictionary,   you get back its keys. That's why that works.  So anyway, I've done it manually here, which is  
a bit sad, but there you go. Okay. So  we can now grab the from train zero,  
which we've already seen. We can grab the x, i.e.  the image, and there it is. There's the image.  
We could grab the first five images  and the first five labels, for example.  
And there they are. Now we already  know what the names of the classes are.  
So we could now see what these map to by grabbing  those features. So there they are. So this is a  
special Hugging Face class, which most libraries  have something including fast.ai that works like  
this. There's something called int to string  {int2str}, which is going to take these and  
convert them to these. So if I call it on our y  batch, you'll see we've got, first is ‘ankle boot’  
and there that is indeed an ankle boot. Now we're  going to have a couple of t-shirts and a dress.  
Okay. So how do we use this to train a model?  Well, we're going to need a DataLoader and we want  
collate function
a DataLoader that for now we're going to do just  like we've done it before. It's going to return,  
well, actually we're going to do something  a bit different. We're going to have,   our collate function is actually going to return  a dictionary. Actually, this is pretty common for  
Hugging Face stuff. And PyTorch  doesn't mind if you, it's happy for   you to return a dictionary from a collation  function. So rather than returning a tuple  
of the stacked up. Hopefully this looks very  familiar. This looks a lot like the thing   that goes through the Dataset for each  one and stacks them up just like we did  
in the previous notebook. So that's what we're  doing. We're doing all in one step here in our  
collate function. And then again, exactly the  same thing. Go through our batch, grab the y  
and this is just stacking them up with the  integers so we don't have to call stack.  
And so we're now going to have the  image and label bits in our dictionary.   So if we create our DataLoader  using that collation function,  
grab one batch. So we can go batch  x dot shape is a 16 by 1 by 28 by 28  
and our y of the batch here, here it is. So the  thing to notice here is that we haven't done any  
transforms or anything or written our own  Dataset class or anything. We're actually  
putting all the work directly in the collation  functions. This is like a really nice way to skip  
all of the kind of abstractions  of your framework, if you want to,  
is you can just do all of your work in  collate functions. So it's going to pass you   each item. So you're going to get the batch  directly. You just go through each item.  
And so here we're saying, okay, grab the x key  from that dictionary, convert it to a tensor  
and then do that for everything in the batch  and then stack them all together. So this is,   yeah, this is like, can be quite a nice way to  do things if you want to do things just very  
manually without having to think too  much about, you know, a framework,   particularly if you're doing really  custom stuff, this can be quite helpful.  
Having said that, Hugging Face datasets  absolutely lets you avoid doing everything  
in collate function, which, if we want  to create really simple applications,   that's where we're going to eventually want  to head. So we can do this using a transform  
transforms function
instead. And so the way we do that is we create  a function. You've got to take our batch. It's  
going to replace the x in our batch with the  tensor version of each of those PIL images.  
And I'm not even stacking them or anything.  And then we're going to return that batch.   And so Hugging Face datasets has something  called with_transform(), and that's going  
to take your dataset, your Hugging Face  dataset, and it's going to apply this function  
to every element. And it doesn't run at all now,  it's going to basically, when, when it, behind  
the scenes, when it calls dunder getitem, it will  call this function on the fly. So, in other words,  
this could have data augmentation, which can  be random or whatever, because it's going to be   rerun every time you grab an item, it's not cached  or anything like that. So other than that, this  
dataset has exactly the API, same API as any other  dataset. It has a length, it has a dunder getitem,  
so you can pass it to a DataLoader. And so PyTorch  already knows how to collate dictionaries of  
tensors. So we've got a dictionary of tensors  now. So that means we don't need a collate   function anymore. I can create a DataLoader from  this without a collate function, as you can see.  
And so this is given exactly  the same thing as before,   but without having to create a custom collate  function. Now, even this is a bit more code  
than I want, having to return this seems a bit  silly. But the reason I had to do this is because   Hugging Face datasets expects the with_transform  function to return the new version of the data.  
So I wanted to be able to write  it like this, transform in place,   and just say the change I want to make and  have it automatically return that. So if  
decorators
I create this function, it's exactly the same  as the previous one, but doesn't have return.  
How would I turn this into something  which does return the result?  
So here's an interesting trick. We could take  that function, pass it to another function to  
create a new function, which is the, a version  of this inplace function that returns the  
result. And the way I do that is by creating a  function called inplace. It takes a function,  
it returns a function. The function it  returns is one that calls my original function  
and then returns the result. So this is the  function. This is a function generating function.  
And it's modifying an inplace function to become  a function that returns the new version of that  
data. And so this is a function. This function is  passed to this function, which returns a function.  
And here it is. So here's the version that Hugging  Face will be able to use. So I can now pass that  
to with_transform() and it  does exactly the same thing.   So this is very, very common in Python. It's so  common that this line of code can be entirely  
removed and replaced with this little token. If  you have a function and put @ at the start, you  
can then put that before a function. And what it  says is take this whole function, pass it to this  
function and replace it with the result. So this  is exactly the same as the combination of this  
and this. And when we do it this way, this kind  of little syntax sugar is called a decorator.  
Okay. So there's nothing magic about decorators.  It's literally, literally identical to this. Oh, I  
guess the only difference is we don't end up with  this unnecessary intermediate underscore version,   but the result is exactly the same. And therefore  I can create a transformed Dataset by using this.  
And there we go. It's all working fine.  
Yeah, so I mean, none of this is particularly  necessary, but what we're doing is we're   just kind of like seeing, you know, the  pieces that we can, we can put in place  
to make this stuff as easy as possible and  that we don't have to think about things   too much. All right. Now with all this, we can  basically make things pretty automatic. And the  
itemgetter
way we can make things pretty automatic is  we're going to use a cool thing in Python   called itemgetter(). And itemgetter is  a function that returns a function. So  
hopefully you're getting used to this idea now.  This creates a function that gets the a and c  
items from a dictionary or something that looks  like a dictionary. So here's a dictionary. It  
contains keys a, b, and c. So this function will  take a dictionary and return the a and c values.  
And as you can see, it has done exactly  that. I’ll explain why this is useful in  
a moment. I just wanted to briefly mention  what did I mean when I said something that   looks like a dictionary? I mean, this is a  dictionary. Okay. That looks like a dictionary.  
But Python doesn't care about what type things  actually are. It only cares about what they look  
like. And remember that when we call something  with square brackets, when we index into  
something, behind the scenes it's just calling  dunder getitem. So we could create our own class.  
And its dunder getitem, gets the key. And it's  just going to manually return 1 if k equals a or  
2 if k equals b or 3 otherwise. And look, that  class also works just fine with an itemgetter.  
The reason this is interesting  is because a lot of people  
write Python as if it's like C++ or Java or  something. They write as if it's this kind  
of statically typed thing. But I really wanted to  point out that it's an extremely dynamic language  
and there's a lot more flexibility than you might  have realized. Anyway, that's a little aside.  
So what we can do is think about a batch for  example where we've got these two dictionaries.  
Okay. So PyTorch comes with a default  collation function called, not surprisingly,  
PyTorch’s default_collate
default_collate So that's part of PyTorch. And  what default_collate() does with dictionaries  
is it simply takes the matching keys and then  grabs their values and stacks them together.  
And so that's why if I call default_collate, a  is now 1, 3, b is now 2, 4. That's actually what  
happened before when we created this DataLoader is  it used the default collation function, which does  
that. It also works on things that are tuples, not  dictionaries, which is what most of you would have  
used before. And what we can do therefore is we  could create something called collate_dict(),  
which is something which  is going to take a Dataset  
and it's going to create a itemgetter function for  the features in that Dataset, which in this case  
is ‘image’ and ‘label’. So this is a function  which will get the ‘image’ and ‘label’ items.  
And so we're now going to return a function  and that function is simply going to call   our itemgetter() on default_collate(). And  what this is going to do is it's going to  
take a dictionary and collate it into a tuple  just like we did up here. So if we run that,  
so we're now going to call DataLoader on our  transform dataset, passing in, and remember,  
this is a function that returns a function.  So it's a collation function for this Dataset  
and there it is. So now this looks a lot like  what we had in our previous notebook. This is   not returning a dictionary, but it's returning  a tuple. So this is a really important idea for,  
particularly, for working with Hugging Face  datasets is that they tend to do things with   dictionaries and most other things in the PyTorch  world tend to work with tuples. So you can just  
use this now to convert anything that takes, that  returns dictionaries into something that provides  
tuples by passing it as a collation function  to your DataLoader. So remember, you know, the  
thing you want to be doing this this week is doing  things like import pdb, pdb.set_trace(), right?  
Put breakpoints, step through, see exactly  what's happening, you know, not just here, but  
also even more importantly, doing it inside the  innermost, inner function. So then you can see,  
what did I do wrong there? Oh,  did I? Set underscore trace.  
So then we can see exactly  what's going on. Put out b.  
List the code. And I could step into it. And  look, I'm now inside the default_collate function,  
which is inside PyTorch. And so I  can now see exactly how that works.  
There it all is. So it's going to go  through and this code is going to look   very familiar because we've implemented all this  ourselves. Because it's being careful to like  
it works for lots of different types of things,  dictionaries, NumPy arrays, so on and so forth.  
Creating a Python library with nbdev
So the first thing I wanted to do, oh,  actually, something I do want to mention here,   this is so useful, we want to be able  to use it in all of our notebooks.  
So rather than copying and pasting this every  time, it would be really nice to create a Python   module that contains this definition. So we've  created a library called nbdev. It's really a  
whole system called nbdev, which does exactly  that. It creates modules you can use from your  
notebooks. And the way you do it is you use this  special thing we call comment directives, which is  
hash pipe. And then hash pipe export. So you put  this at the top of a cell and it says do something  
special for this cell. What this does is it says  put this into a Python module for me, please.  
Export it to a Python module. What Python  module is it going to put it in? Well,   if you go all the way to the top, you tell it what  default export module to create. So it's going to  
create a module called datasets. So what I do at  the very end of this module is I've got this line  
that says import nbdev, nbdev.nbdev_export().  And what that's going to do for me is create  
a library, a Python library. It's going to have  a datasets.py in it. And we'll see everything  
that we exported. Here it is. collate_dict  will appear in this for me. And so what that  
means is now in the future, in my notebooks,  I will be able to import collate_dict from  
my datasets. Now you might wonder, well, how  does it know to call it miniai? What's miniai?  
Well, in nbdev, you create a settings.ini file  where you say what the name of your library is.  
So we're going to be using this quite a lot  now because we're getting to the point where  
we're starting to implement stuff that didn't  exist before. So previously most of the stuff,  
or pretty much all the stuff we've created, I've  said like, oh, that already exists in PyTorch.  
So we don't need it. We just use PyTorch’s. But  we're now getting to a point where we're starting  
to create stuff that doesn't exist anywhere. We've  created it ourselves. And so therefore we want to  
be able to use it again. So during the rest of  this course, we're going to be building together  
a library called miniai That's going to be our  framework, our version of something like fastai.  
Maybe it's something like what fastai 3 will end  up being. We'll see. So that's what's going on  
here. So we're going to be using, once I  start using miniai, I'll show you exactly  
how to install this, but that's what this  export is. And so you might've noticed I   also had an export on this in place thing. And  I also had it on my necessary import statements.  
Plotting images
Okay. We want to be able to see what this dataset  looks like. So I thought it now is a good time to  
talk a bit about plotting because knowing how  to visualize things well is really important.  
And again, the idea is we, we're not allowed  to use fastai's plotting library. So we've got  
to learn how to do everything ourselves. So  here's the basic way to plot an image using  
matplotlib. So we can create a batch, grab the  x part of it, grab the very first thing in that.  
And imshow() means show an image. And  here it is. There's our ankle boot.  
So let's start to think about what stuff we  might create, which we can export to make  
this a bit easier. So let's create something  called show_image(), which basically does  
imshow(), but we're going  to do a few extra things.   We will make sure that it's in the correct  access order. We will make sure it's not on  
CUDA that's on the CPU. If it's not a NumPy  array, we'll convert it to a NumPy array.  
We'll be able to pass in an existing axis,  which we'll talk about soon. If we want to,   we'll be able to set a title if we want to.  Amd also, this thing here removes all this  
ugly 05 blah blah blah axis because we're  showing an image. We don't want any of that.  
So if we try that, you can see, there we go. We  also been able to say what size we want the image.  
There it all is. Now here's something  interesting. When I say help,  
the help shows the things that I implemented,  but it also shows a whole lot more things.  
How did that magic thing happen? And you  can see they work because here's figsize,   which I didn't add. Oh, sorry. I did add.  Well, okay. That's a bad example. Anyway,  
these other ones all work as well. So how did  that happen? Well, the trick is that I added  
kwargs and fastcore’s delegates
**kwargs here and **kwargs says, grab, you  can pass it as many or any other arguments  
as you like that aren't listed. And they'll  all be put into a dictionary with this name.  
And then, when I call imshow() I pass that entire  dictionary ** here means “as separate arguments”.  
And that's how come it works. And then  how come does it know, how come it knows   what help to provide? The reason why is that  fastcore has a special thing called delegates,  
which is a decorator. So now you know  what a decorator is and you tell it,  
what is it that you're going to be passing kwargs  to? I'm going to be passing it to imshow(), and  
then it automatically creates the documentation  correctly to show you what kwargs can do.  
So this is a really helpful way of being able to  kind of extend existing functions like imshow and  
still get all of their functionality and  all of their documentation and add your   own. So delegates is one of the most useful  things we have in fastcore, in my opinion. So  
we're going to export that. So now we can use  show_image() anytime we want, which is nice.  
Something that's really helpful to  know about matplotlib is how to create  
subplots. So for example, what happens if you  want to plot two images next to each other?  
So in matplotlib subplots creates multiple  plots and you pass it number of rows and the  
number of columns. So this here has,  as you see, one row and two columns.  
And it returns axes. Now what it calls axes  is what it refers to as the individual plots.  
So if we now call show_image() on  the first image, passing in axs[0],  
it's going to get that here, right? Then we  call ax.imshow(). That means put the image  
on this subplot. They don't call it a subplot,  unfortunately, they call it an axis, put it on   this axis. So that's how come we're able to  show an image, one image on the first axis,  
and then show a second image on the second axis by  which we mean subplot. And there's our two images.  
So that's pretty handy. So I've decided to add  some additional functionality to subplots. So  
therefore I use delegates on subplots() because  I'm adding functionality to it. And I'm going  
to be taking kwargs and passing it through to  subplots(). And the main thing I wanted to do  
is to automatically create an appropriate figure  size by just finding out, you tell us what image  
size you want. And I also want to be able to  add a title for the whole set of subplots.  
And so there it is. And then I also want  to show you that it'll automatically,  
if we want to, create documentation for us  as well, for our library. And here is the  
documentation. So as you can see here, for the  stuff I've added, it's telling me exactly what  
each of these parameters are, their type,  their defaults, and information about each one.  
And that information is automatically coming from  these little comments. We call these documents.  
This is all automatic stuff done by fastcore and  nbdev. And so you might've noticed when you look  
at fastai library documentation, it always has  all this info. So that's why. You don't actually  
have to call show_doc(), it automatically added to  your documentation for you. I'm just showing you  
here what it's going to end up looking like. And  you can see that it's worked with delegates. It's   put all the extra stuff from delegates in here  as well. And here they are all listed out here  
as well. So anyway, subplots. So let's create  a 3 by 3 set of plots and we'll grab the first  
eight images. And so now we can go through each  of the subplots. Now it returns it as a 3 by 3,  
basically a list of 3 lists of 3 items. So  I flattened them all out into a single list.  
So we'll go through each of those subplots and  go through each image and show each image on  
each axis. And so here's a quick way to quickly  show them all. As you can see, it's a little bit  
ugly here, so we'll keep on adding more useful  plotting functionality. So here's something that,  
again, it calls our subplots delegates to it.  But we're going to be able to say, for example,  
how many subplots do we want? And it'll  automatically calculate the rows and the columns.  
And it's going to remove the axes for any ones  that we're not actually using. And so here we got  
that. So that's what get_grid()'s going to let us  do. So we're getting quite close. And so, finally,  
why don't we just create a single thing called  show_images() that's going to get our grid.  
And it's going to go through our images optionally  with a list of titles and show each one.  
And we can use that here. You can see we have  successfully got all of our labeled images.  
And so we, yeah, I think all this stuff for the  plotting is pretty useful. So as you might've  
noticed, they were all exported. So in our  datasets.py, we've got our get_grid(), we've  
got our subplots, we've got our show_images().  So that's going to make life easier for us now,   since we have to create everything from  scratch, we have created all of those things.  
So as I mentioned at the very end,  we have this one line of code to run.   And so just to show you, if I remove  
miniai dot datasets… miniai slash datasets.py, so  it's all empty. And then I run this line of code.  
And now it's back, as you can see, and it  tells you it's auto generated. All right. So  
Computer Science concepts with Python: callbacks
we are nearly at the point where we can build  our learner. And once we've built our learner,   we're going to be able to really dive deep into  training and studying models. So we've kind of  
got, nearly got all of our infrastructure in  place. Before we do, there's some pieces of  
Python, which not everybody knows, and I want  to kind of talk about and kind of computer   science concepts I want to talk about.  So that's what 06_foundations is about.  
So this whole section is just going to tell it,  just going to talk about some stuff in Python  
that you may not have come across before. Or maybe  it's a review for some of you as well. And it's  
all stuff we're going to be using basically in the  next notebook. So that's why I wanted to cover it.  
So we're going to be creating a learner class.  So a learner class is going to be a very general  
purpose training loop, which we can get to do  anything that we want it to do. And we're going  
to be creating things called callbacks to make  that happen. And so therefore we're going to   just spend a few moments talking about what are  callbacks, how are they used in computer science,  
how are they implemented, look at some examples.  They come up a lot. That's the most common place  
that you see callbacks in software is for GUI  events. So for events from some graphical user  
interface. So the main graphical user interface  library in Jupyter Notebooks is called ipywidgets.  
And we can create a widget like a button, like  so. And when we display it, it shows me a button.  
And at the moment it doesn't  do anything if I click on it.  
What we can do though, is we can  add an on_click() callback to it,  
which is something which is a fun,  we're going to pass it a function,   which is called when you click it. So let's  define that function. So I'm going to say  
w.on_click(f) is going to assign the f function  to the on click callback. Now if I click this,  
there you go, it's doing it. Now what does that  mean? Well, a callback is simply a callable that  
you've provided. So remember a callable is a more  general version of a function. So in this case,   it is a function that you've provided that will  be called back to when something happens. So in  
this case, there's something that's happening is  that they're clicking a button. So this is how we  
are defining and using a callback as a GUI event.  So basically everything in ipywidgets, if you want  
to create your own graphical user interfaces  for Jupyter, you can do it with ipywidgets and  
by using these callbacks. So these particular  kinds of callbacks are called events, but it's  
just a callback. All right, so that's somebody  else's callback. Let's create our own callback.  
So let's say we've got some very slow calculation.  And so it takes a very long time to add up  
the numbers zero to five squared because  we sleep for a second after each one. So   let's run our slow calculations. Still  running. Oh, how's it going? Come on,  
finish our calculation. There we go. The answer  is 30. Now for a slow calculation like that,   such as training a model, it's a slow calculation.  It would be nice to do things like, I don't know,  
print out the loss from time to time  or show a progress bar or whatever.   So generally for those kinds of things, we would  like to define a callback that is called at the  
end of each epoch or batch or every few seconds or  something like that. So here's how we can modify  
our slow calculation routine such that you can  optionally pass at a callback. And so all of  
these codes are the same, except we've added this  one line of code that says, if there's a callback,  
then call it and pass in where we're up to. So  then we could create our callback function. So  
this is just like we created a full callback  function f(), let's create a show_progress()   callback function. That's going to tell us how far  we've got. So now if we call show slow calculation  
passing in our callback, you can see it's going  to call this function at the end of each step.  
So here we've created our own callback. So  there's nothing special about a callback. It  
doesn't require its own like syntax. It's not  a new concept. It's just an idea, really, which  
is the idea of passing in a function, which some  other function will call at particular times, such  
as at the end of a step or such as when you click  a button. So that's what we mean by callbacks.  
We don't have to define  the function ahead of time.   We could define the function at  the same time that we call the  
Lambdas and partials
slow calculation by using Lambda. So as we've  discussed before, Lambda just defines a function,  
but it doesn't give it a name. So here's a  function that takes one parameter and prints   out exactly the same thing as before. So here's  the same way as doing it, but using a Lambda.  
We could make it more sophisticated  now. And rather than always saying,   “Awesome! We finished epoch…”, whatever, we  could have let you pass in an exclamation  
and we print that out. And so in this case, we  could now have our Lambda call that function.  
And so one of the things that  we can do now is to, again,   we can create a function that returns a function.  
And so we could create a make_show_progress  function where you pass in the exclamation.  
We could then create, and there's no need to give  it a name actually, it's just return it directly.  
We can return a function that  calls that exclamation. So here we  
are passing in nice. And that's exactly the same  as doing something like what we've done before.  
We could say, instead of using a Lambda,  we can create an inner function like this.  
So here's now a function that returns a  function. This does exactly the same thing.  
Okay. So one way with the Lambda,  one way with outer Lambda.  
One of the reasons I wanted to show you  that is so I can, I don't know about…  
so many here, is that we can do exactly the  same thing using partial. So with partial,  
it's going to do exactly the same thing  as this kind of make_show_progress().   It's going to call show_progress() and  pass, okay, I guess. So this is again,  
an example of a function returning a function.  And so this is a function that calls show progress  
passing in this as the first parameter.  And again, it does exactly the same thing.  
Okay. So we tend to use partial a lot.   So that's certainly something worth spending  time practicing. Now as we've discussed,  
Python doesn't care about types in particular.  And there's nothing about any of this  
Callbacks as callable classes
that requires cb to be a function. It just has to  be a callable. A callable is something that you  
can call. And so as we've discussed, another way  of creating a callable is defining dunder call.  
So here's a class and this is going to work  exactly the same as our make show progress thing,  
but now as a class. So there's a dunder init,  which stores the exclamation and a dunder call,  
the prints. And so now we're creating a object  which is callable and does exactly the same thing.  
Okay. So these are all fundamental ideas that  I want you to get really comfortable with. The  
idea of dunder call, dunder things in general,  partials, classes, because they come up all  
the time in PyTorch code and in the code we'll be  writing and, in fact, pretty much all frameworks.  
So it's really important to feel comfortable  with them. And remember you don't have to rely on  
the resources we're providing. If there are  certain things here that are very new to you,  
Google around for some tutorials or ask for  help on the forums, finding things and so forth.  
Multiple callback funcs; *args and **kwargs
And then I'm just going to briefly  recover something I've mentioned before,   which is *args and **kwargs,  because again, they come up a lot.  
I just wanted to show you how they work. So if  we create a function that has *args and **kwargs,  
nothing else, and I'm just going to  have this function just print them.  
Now I'm going to call the function. I'm going  to pass 3. I'm going to pass “a” and I'm going  
to pass thing1=”hello”. Now these are parts, what  we would say, by position. We haven't got a blah  
equals. They're just stuck there. Things that are  passed by position are placed in *args, if you  
have one, it doesn't have to be called args. You  can call this anything you like, but in the star   bit. And so you can see here that args is a tuple  containing the positionally passed arguments. And  
then kwags is a dictionary containing the named  arguments. So that is all that *args and **kwargs  
do. And as I say, there's nothing special about  these names. I'll call this a, I'll call this b.  
Okay. And it'll do exactly the same  thing. Okay. So this comes up a lot.  
And so it's important to remember that this  is literally all that they're doing. And then,  
on the other hand, let's say we had  a function which takes a couple of,  
okay, let's try that, print a, actually,  we'll just print them directly a, b, c. Okay.  
We can also, rather than just using them as  parameters, we can also use them when calling  
something. So let's say I create something called  args, again, it doesn't have to be called args,   called, which contains [1, 2]. And I create  something called kwags that contains a dictionary  
containing {‘c’: 3}. I can then call g()  and I can pass in *args comma **kwargs.  
And that's going to take this 1, 2,  and pass them as individual arguments,  
positionally. And it's going to take the {‘c’:  3} and pass that as a named argument, c equals 3.  
And there it is. Okay. So there are two  linked but different ways that use * and **.  
Okay. Now here's a slightly different way  of doing callbacks, which I really like.  
In this case, I've now passing in  a callback that's not callable,   but instead it's going to have a method called  before_calc and another method called after_calc.  
And I'm, so now my callback is going to be a class  containing a before_calc and an afte_calc method.  
And so if I run that, you can see it's… that there  it goes. Okay. And so this is printing before  
and after every step by calling before_calc() and  after_calc(). So callback actually doesn't have to  
be a callable. It doesn't have to be a function. A  callback could be something that contains methods.  
So we could have a version of this,  which actually, as you can see here,  
it's going to pass in to after_calc(), both  the epoch number and the value it's up to,  
but by using *args and **kwags, I can just  safely ignore them if I don't want them.  
Right. So it's just going to chew them up and  not complain. If I didn't have those here,  
it won't work. See, because  it got passed in val equals  
and there's nothing here looking for  val equals. And it doesn't like that.  
So this is one good use of *args and **kwags  is to eat up arguments you don't want.  
Or we could use the arguments. So let's  actually use epoch and val and print them out.  
And there it is. So this is a more sophisticated  callback that's giving us status as we go.  
Skip this bit because we don't really care about  that. Okay. So finally, let's just review this  
__dunder__ thingies
idea of dunder, which we've mentioned before,  but just to really nail this home, anything that  
looks like this, underscore underscore something  underscore underscore something is special. And  
basically it could be that Python has to find that  special thing or PyTorch has to find that special   thing or NumPy has to find that special thing, but  they're special. These are called dunder methods.  
And some of them are defined as part of the  Python data model. And so if you go to the Python  
documentation, it'll tell you about these various  different— here's __repr__, which we used earlier.  
Here's __init__ that we used earlier. So  they're all here. PyTorch has some of its own,  
NumPy has some of its own. So for example, if  Python sees plus (+), what it actually does  
is it calls dunder add. So if we want to create  something that's not very good at adding things,  
it actually always adds 0.01 to it.  
Then I can say SloppyAdder(1) + SloppyAdder(2)  equals 3.01. So “+” here is actually calling  
dunder add. So if you're not familiar with  these, click on this data model link and read  
about these specific one, two, three, four,  five, six, seven, eight, nine, ten, eleven  
methods, because we'll be using all of these  in the course. So I'll try to revise them when  
we can, but I'm generally going to assume that  you know these. A particularly interesting one  
is getattr. We've seen setattr already. getattr  is just the opposite. Take a look at this.  
Here's a class. It just contains two attributes,  a and b, that are set to 1 and 2. So I'll create  
an object of that class a.b equals 2, because I  set b to 2. Okay. Now when you say a.b, that's  
just syntax sugar basically, in Python. What it's  actually calling behind the scenes is getattr.  
It calls getattr on the object. And so this  one here is the same as getattr(a, ‘b’),  
which hopefully, oh, actually that'll be, yeah,  so it calls getattr(a, ‘b’). And this can kind of  
be fun because you could call getattr a, and then  either ‘b’ or ‘a’ randomly. How's that for crazy?  
So if I run this, 2, 1, 1, 1, 2, as you can see,  it's random. So yeah, Python is such a dynamic  
language. You can even set it up so you literally  don't know what attributes are going to be called.  
Now getattr, behind the scenes, is actually  calling something called dunder getattr. And by  
default, it'll use the version in the object base  class. So here's something just like a, it's got  
a and b defined, but I've also got dunder getattr  defined. And so dunder getattr, it's only called  
for stuff that hasn't been defined yet, and it'll  pass in the key or the name of the attribute.  
So generally speaking, if the first character  is an underscore, it's going to be private or  
special. So I've just got to raise an  attribute error. Otherwise I'm going to  
steal it and return f‘Hello from {k}’. So if I  go b.a, that's defined. So it gives me 1. If I go  
b.foo, that's not defined. So it calls getAtra and  I get back hello from foo. And so, this gets used  
a lot in both fastai code and also a Hugging Face  code to often make it more convenient to access  
things. So that's, yeah, that's how the getattr  function and the dunder getattr method work.  
Wrap-up
Okay. So I went over that pretty quickly. Since  I know for quite a few folks, this will be all  
review, but I know for folks who haven't seen  any of this, this is a lot to cover. So I'm  
hoping that you'll kind of go back over this,  revise it slowly, experiment with it and look   up some additional resources and ask on the forum  and stuff for anything that's not clear. Remember,  
everybody has parts of the course that's really  easy for them and parts of the course that are  
completely unfamiliar for them. And so  if this particular part of the course is   completely unfamiliar to you, it's not because  this is harder or going to be more difficult  
or whatever. It's just so happens that this is  a bit that you're less familiar with, or maybe  
the stuff about calculus in the last lesson was  a bit that you're less familiar with. There isn't  
really anything particularly in the course that's  more difficult than other parts. It's just that,  
you know, based on whether you happen to have  that background. And so, yeah, if you spend   a few hours studying and practicing, you know,  you'll be able to pick up these things. And yeah,  
so don't stress if there are things that you don't  get right away. Just take the time. And if you,  
yeah, if you do get lost, please ask because  people are very keen to help. If you've   tried asking on the forum, hopefully you've  noticed that people are really keen to help.  
All right. So, I think this has been a pretty  successful lesson. We've got to a point where  
we've got a pretty nicely optimized training  loop. We understand exactly what DataLoaders   and Datasets do. We've got an optimizer. We've  been playing with Hugging Face datasets. And  
we've got those working really smoothly. So we  really feel like we're in a pretty good position  
to write our generic learner training loop and  then we can start building and experimenting with  
lots of models. So look forward to seeing you  next time to doing that together. Okay. Bye.

---

# 15

Hi all and welcome to Lesson 15. And what  we're going to endeavor to do today is to  
create a convolutional autoencoder.  And in the process, we will see why  
doing that well is a tricky thing to do and  time permitting, we will begin to work on a  
framework, a deep learning framework  to make life a lot easier. Not sure  
how far we'll get on that today time wise. So  let's see how we go and get straight into it.
Okay. So today, let's start by talking before  we can create a convolutional autoencoder,  
we know to talk about convolutions and what are  they and what are they for. Broadly speaking,  
What are convolutions?
convolutions are something that allows us to, to  tell our neural network a little bit about the  
structure of the problem. That's going to make  it a lot easier for it to solve the problem. And in particular, the structure of our problem  is we're doing things with images. Images are  
laid out on a grid, a 2d grid for black and white  or a 3d for color or a 4d for a color video or  
whatever. And so we would say, you know, there's  a relationship between the pixels going across  
and the pixels going down. They tend to be similar  to each other, differences in those pixels across  
those dimensions tend to have meaning. Sets,  patterns of pixels that appear in different  
places often represent the same thing. So for  example, a cat in the top left is still a cat,  
even if it's in the bottom right. These kinds  of, this kind of prior information is something  
that is naturally captured by a Convolutional  Neural Network, something that uses convolutions.
Generally speaking, this is a good thing  because it means that we will be able to   use less parameters and less computation because  more of that information about the problem we're  
solving is kind of encoded directly into our  architecture. There are other architectures that  
don't encode that prior information as strongly,  such as a Multi-Layer Perceptron, which we've been  
looking at so far, or a Transformers network,  which we haven't looked at yet. Those kinds of  
architectures could potentially give us, or they  do give us more flexibility and given enough time,  
compute and data, they could potentially find  things that maybe CNNs would struggle to find.
So we're not always going to use  Convolutional Neural Networks,   but they're a pretty good starting point and  certainly something important to understand.  
They're not just used for images. We can also  take advantage of one-dimensional convolutions  
for language-based tasks, for instance.  So convolutions come up a lot.
So in this notebook, one thing you'll  notice that might be of interest is we  
are importing stuff from miniai now. Now  miniai is this little library that we're  
starting to create and we're creating it  using nbdev. So we've got a miniai.training  
and a miniai.datasets. And so if we look,  for example, at the datasets notebook,  
it starts with something that says that the  default export is called datasets. And some  
of the cells have a export directive on them.  And at the very bottom, we had something that  
called nbdev_export(). Now what that's going to do  is it's going to create a file called datasets.py  
just here, datasets.py. And it contains,  
those cells that we exported.   And why does it, why is it called miniai.datasets?  That's because everything for nbdev is stored  
in settings.ini. And there's something here  saying create a library libname called miniai.  
You can't use this library until you install it.  Now we haven't uploaded it to PyPy, like made it  
a pip installable package from a public server.  But you can actually install a local directory  
as if it's a Python module that you've kind of  installed from the internet. And to do that,  
you say pip install, in the usual way, but  you say -e, that stands for editable. And  
that means set up the current directory as  a Python module. Well, current directory,   actually any directory you like, I just put  dot to mean the current directory. And so  
you'll see that's going to go ahead and actually  install my library. And so after I've done that,  
I can now import things from  that library as you see.
Okay. So this is just the same as before. We're  going to grab our MNIST dataset and we're going   to create a Convolutional Neural Network on  it. So before we do that, we're going to talk  
about what are convolutions. And one of my  favorite descriptions of convolutions comes  
from a student in our, I think it was our very  first course, Matt Kleinsmith, who wrote this  
really nice Medium article, CNNs from different  viewpoints, which I'm going to steal from. And  
here's the basic idea. Say that this is our image.  It's a 3 by 3 image with 9 pixels labeled from A  
to J as capital letters. Now a convolution  uses something called a kernel and a kernel  
is just another tensor. In this case, it's a 2  by 2 matrix again. So it's, I mean, this one's,  
we're going to have alpha, beta, gamma,  delta, alpha, beta, gamma, delta as our four  
values in this convolution. Now in this  kernel, oh, now one thing I'll mention,  
I can't remember if I've said this before, is the  Greek letters are things that you want to be able  
to, I think I have mentioned this, you want to  be able to pronounce them. So if you don't know   how to read these and say what these names are,  make sure you head over to Wikipedia or whatever,  
and learn the names of all the Greek letters so  that you can, cause they come up all the time.
Okay. So what happens when we apply a convolution  with this 2 by 2 kernel to this 3 by 3  
Visualizing convolutions
image? I mean, it doesn't have to be an image  it's in this case, it's just a rank 2 tensor,  
but it might represent an image. What happens  is we take the kernel and we overlay it over  
the first little 2 by 2 sub grid, like so. And  specifically what we do is we match color to  
color. So the output of this first 2 by 2  overlay would be alpha times a plus beta  
times b plus gamma times d plus delta times  e, and that would yield some value P, and  
that's going to end up in the top left of a 2 by  2 output. So the top right of the 2 by 2 output,  
we're going to slide, it's like a sliding window,  we're going to slide our kernel over to here  
and apply each of our coefficients to  these respectively colored squares.  
And then ditto for the bottom left  and then ditto for the bottom right.
So we end up with this equation. P, as  we discussed, is alpha a plus beta b  
plus gamma d plus delta e, plus some bias  term. Q, to the top right, as you can see,  
it's just alpha in this test times b. And  so we just take multiplying them together  
and adding them up, multiply together, add  them up, multiply together and add them up.   So we're basically, you can imagine that  we're basically flattening these out into  
rank 1 tensors into vectors and then doing a dot  product would be one way of thinking about what's   happening as we slide this kernel over these  windows. And so this is called a convolution.
So let's try and create a convolution. So,  for example, let's grab our training images.  
Creating a convolution with MNIST
And take a look at one.   And let's create a 3 by 3 kernel. So remember,  a kernel is just ‒we've already‒ kernel appears  
a lot of times in computer science and math.  We've already seen the term kernel to mean  
a piece of code that we run on a GPU across lots  of parallel kind of virtual devices or potentially  
in a grid. There's a similar idea here. We've  got a computation, which is in this case, kind of  
this dot product or something like a dot product  sliding over, occurring lots of times over a grid.  
But it's yeah, it's a bit different.  That's kind of another use of the word  
kernel. So in this case, a kernel is a, in  this case, it's going to be a rank 2 tensor.   And so let's create a kernel with these values  in the 3 by 3 matrix, rank 2 tensor. And we could  
draw what that looks like. Not surprising,  it just looks like a bunch of lines. Oops.
OK, so what would happen if we slide this over  just these nine pixels over this 28 by 28?  
Well, what's going to happen is  if we've got some‒ the top left,   for example, 3 by 3 section has these names,  then we're going to end up with negative a1,  
because the top three are all negative. Right?  Negative a1, minus a2, minus a3. The next are just  
zero. So that won't do anything. And then plus  a7, plus a8, plus a9. Why is that interesting?  
That's interesting. Well, let's try  here. What I've done here is I've grabbed  
just the first 13 rows and first 23 columns of  our image, and I'm actually showing the numbers  
and also using gray kind of conditional  formatting, if you like, or the equivalent in  
pandas to show this top bit. So we're looking at  just this top bit. So what happens if we take rows  
3, 4 and 5? Remember, this is not inclusive,  right? So it's rows 3, 4 and 5, columns 14, 15,  
16… 14, 15, 16. So we're looking at this, these  three here. What's that going to give us if we  
multiply it by this kernel? It gives us a fairly  large positive value because the three that we  
have negatives on is the top row. Well, they're  all zero. And the three that we have positives on,  
they're all close to 1. So we end up with  quite a large number. What about the same  
columns but for rows 7, 8, 9? 7, 8, 9. Here, the  top is all positive and the bottom is all zero.  
So that means that we're going to get a lot of  negative terms. And not surprisingly, that's  
exactly what we see. If we do this, kind of a dot  product equivalent, which all you need in NumPy to  
do that is just an element-wise multiplication  followed by a sum, right? So that's going to be  
quite a large negative number. And so perhaps  you're seeing what this is doing. And maybe you   got a hint from the name of the tensor we created.  It's something that is going to find the top edge.  
So this one is a top edge, so it's a positive.  And this one is a bottom edge, so it's a negative.
So we would like to apply that, this kernel,  to every single 3 by 3 section in here.  
So we could do that by creating a little  apply_kernel function that takes some particular  
row and some particular column and some particular  tensor as a kernel and does that multiplication  
.sum() that we just saw. So  for example, we could replicate  
this one by calling apply_kernel(). And this  here is the center of that 3 by 3 grid area.  
And so there's that same number, 2.97. So now  we could apply that kernel to every one of the  
3 by 3 sections. 3 by 3 windows in this 28 by 28  image. So we're going to be sliding over like this  
red bit sliding over here. But we've actually  got a 28 by 28 input, not just a 5 by 5 input.  
So to get all of the coordinates, let's just  simplify it to do 5 five by 5. We can go,  
we can create a list comprehension. We can take  i through every value in range(5). And then for  
each of those, we can take j for every value in  range(5). And so if we just look at that tuple,  
you can see we get a list of lists containing  all of those coordinates. So this is a list  
comprehension in a list comprehension, which when  you first say it may be surprising or confusing,  
but it's a really helpful idiom. And I  certainly recommend getting used to it.
Now, what we're going to do is we're not just  going to create this tuple, but we're actually  
going to call apply_kernel() for each of those. So  if we go through from 1 to 27, well, actually 1 to  
26, because 27 is exclusive. So we're going to go  through everything from 1 to 26. And then for each  
of those, go through from 1 to 26 again and call  apply_kernel(). And that's going to give us the  
result of applying that convolutional kernel to  every one of those coordinates. And there's the  
result. And you can see what it's done as we  hoped is it is highlighting the top edges. So  
yeah, you might find that kind of surprising  that it's that easy to do this kind of image  
processing. We're literally just doing an element  wise multiplication and a sum for each window.
Okay, so that is called a convolution. So we  can do another convolution. This time we could  
do one with the left edge tensor. So as you  can see, it looks just a rotated version or  
transposed version, I guess, of our top  edge tensor, here's what it looks like.   And so if we apply that kernel, so this time  we're going to apply the left edge kernel.
And so notice here that we're  actually passing in a function,   right? We're passing in a function. Sorry, not  a function, is it? It's just a tensor actually.  
So we're going to pass in the left edge tensor   for the same list comprehension, in  a list comprehension. And this time  
we're getting back the left edges. It's  highlighting all of the left edges in the digit.  
So yeah, this is basically what's happening here  is that a 2 by 2 can be looped over an image,  
creating these outputs. Now you'll see here that in the process  of doing so, we are losing the outermost  
pixels of our image. We'll learn about  how to fix that later. But just for now,   notice that as we are putting our 3 by 3 through,  for example, in this 5 by 5 , there's only one,  
two, three places that we can put it going across,  not five places because we need some kind of edge.
All right, so that's cool. That's  a convolution. And hopefully if you   remember back to kind of the Zeiler  and Fergus pictures from Lesson 1,  
you might recognize that the kind of first layer  of a convolutional network is often looking for  
kind of edges and gradients and things like  that. And this is how it does it. And then   the convolutions on top of convolutions  with nonlinear activations between them  
can combine those into curves, or corners  or stuff like that, and so on and so forth.
Speeding up the matrix multiplication when calculating convolutions
Okay, so how do we do this quickly? Because  currently this is going to be super, super slow   doing this in Python. So one of the very earliest  or probably the earliest publicly available  
general purpose deep learning, GPU accelerated  deep learning thing I saw was called Caffe.  
That was created by somebody called Yangqing Jia.  And he actually described what happened, how Caffe  
went about implementing a fast convolution on a  GPU. And basically he said, well, I had two months  
to do it and I had to finish my thesis. And so I  ended up doing something where I said, well, there  
was some other code out there, Krizhevsky, who you  might have come across, him and Hinton set up a  
little startup which Google bought, and that kind  of became the start of Google's deep learning,  
Google Brain, basically. And so Krizhevsky  had all this fancy stuff in his library,  
but Yangqing Jia said, oh, I didn't know how to  do all that stuff. So I said, well, I already know  
how to multiply matrices, so maybe I can convert  a convolution into a matrix multiplication.
And so that became known as im2col.  
im2col is a way of converting a convolution  into a matrix multiply. And so actually,  
I don't know if, I suspect Yangqing Jia kind of  accidentally reinvented it, because it actually   had been around for a while, even at the point  that he was writing his thesis, I believe.  
So it was actually, this is the place  I believe it was created in this paper.  
So that was in 2006, which is a while ago.  And so this is actually from that paper.  
And what they describe is, let's  say you are putting this 2 by 2  
kernel over this 3 by 3 bit of an image. So  here you've got this window needs to match to  
this bit of this window. What you could  do is you could unwrap this to 1, 2, 1,  
2 downwards to here. 1, 2, 1, 2, to unroll it  like so. And you could unroll the kernel here.  
Yeah, so this is 1, 2, 1, 1. So this bit is here,  1, 2, 1, 1. And then you could unroll the kernel  
1, 1, 2, 2 to here, 1, 1, 2, 2. And then once  they've been flattened out and moved in that way,  
and then you'll do exactly the same  thing for this next patch here, 2,   0, 1, 3. You flatten it out and put it here, 2,  0, 1, 3. So if you basically take those kernels  
and flatten them out in this format, then you  end up with a matrix multiply. If you multiply  
this matrix by this matrix, you'll end up with  the output that you want from the convolution.
So this is basically a way of unrolling   your kernels and your input features into  matrices, such as when you do the matrix multiply,  
you get the right answer. So it's a kind of  a nifty trick. And so that is called im2col.
I guess we're kind of cheating a little bit.  Implementing that is kind of boring. It's just   a bunch of copying and tensor manipulation. So I  actually haven't done it. Instead, I've linked to  
a NumPy implementation, which is here. And  it also, part of it is this get_indices(),  
which is here. And as you can see, it's a little  bit tedious with repeats and tiles and reshapes  
and whatnot. So I'm not going to call it homework,  but if you want to practice your tensor indexing  
manipulation skills, try creating a PyTorch  version from scratch. I got to admit, I didn't  
Pythorch’s F.unfold and F.conv2d
bother. Instead, I used the one that's built  into PyTorch. And in PyTorch, it's called unfold.
So if we take our image and PyTorch expects  there to be a batch axis and a dimension and a  
channel dimension. So we'll add two unit leading  dimensions to it. Then we can unfold our input  
for a 3 by 3. And that will give us a 9 by  676 input. And so then we can take that…  
and then we'll take our kernel and just flatten it  out into a vector. So view() changes the shape and  
-1 just says dump everything into this dimension.  So that's going to create a 9 long vector, length  
9 vector. And so now we can do the matrix multiply  just like they've done here of the kernel matrix.
That's our weights by the unrolled input features.  
And so that gives us a 676 long. We can then view  that as 26 by 26 and we get back as we hoped our  
left edge tensor result. And so this is how  we can kind of from scratch create a better  
implementation of convolutions. The reason I'm  cheating, I'm allowed to cheat here is because  
we did actually create convolutions from  scratch. We're not always creating the GPU   optimized versions from scratch, which was never  something I promised. So I think that's fair. 
But it's cool that we can kind of hack it  out a GPU optimized version in the same   way that the kind of original deep learning  library did. So if we use apply_kernel(),  
we get nearly nine milliseconds. If  we use unfold() with matrix multiply,  
we get 20 microseconds. So that's what about  400 times faster. So that's pretty cool.
Now, of course, we don't have to use  unfold() and matrix multiply because  
PyTorch has a conv2d(). So we can run that.  And that interestingly is about the same speed,  
at least on GPU. But this would  also work on GPU just as well.  
Yeah, I'm not sure this will always be the  case. In this case, it's a pretty small   image. I haven't experimented a whole lot  to see whereabouts there's a big difference  
in speeds between these. Obviously, I always  just use F.conv2d. But if there's some more  
tricky convolution you need to do with some weird  thing around channels or dimensions or something,  
you can always try this unfold trick.  It's nice to know it's there, I think.
So we could do the same thing for diagonal edges.   So here's our diagonal edge  kernel or the other diagonal.  
So if we just grab the first 16 images.  
Then we can do a convolution on our whole  batch with all of our kernels at once.  
So this is a nice optimized thing that  we can do. And you end up with your 26  
by 26. You've got your 4 kernels and you've got  your 16 images. And so that's summarized here.
So that's generally what we're doing to get  good GPU acceleration is we're doing a bunch   of kernels and a bunch of images all at once  across all of their pixels. And so here we go.  
That's what happens when we take a look at  our various kernels for a particular image.  
Left edge, I guess top edge and then  diagonal top left and top right.  
OK, so that is optimized convolutions  on… and that works just as well on CPU  
or GPU. Obviously, GPU will  be faster if you have one. Now, how do we deal with the  problem that we're losing  
Padding and Stride
one pixel on each side? What we can do  is we can add something called padding.  
And for padding, what we basically do  is rather than starting our window here,   we start it right over here and actually would be  up one as well. And so these 3 on the left here.  
We just take the input for each of those  as 0. So we're basically just assuming  
that they're all zero. I mean, there's  other options we could choose. We could   assume they're the same as the one next  to them. There's various things we can do,  
but the simplest and the one we normally  do is just assume that there's zero. So now. So let's say, for example, this  is, this is called one pixel padding.  
Let's say we did 2 pixel padding. So we had  2 pixel padding with a 5 by 5 input. OK,  
and a 4 by 4 kernel. So that grays our kernel.  Right, then we're going to start right up way  
over here on the corner. OK, and then you can see  what happens as we slide. The kernel over, there's  
all the spots that it's going to take, and so that  this dotted line area is the area that we're kind  
of effectively going through, but all of these  white bits, we're just going to treat as zero.   And so and then this is this green is the output  size we end up with, which is going to be 6 by 6.  
For a 5 by 5 input, I should mention.
Even numbered edge kernels are not used  very often, we normally used odd numbered   kernels if you use, for example, a 3  by 3 kernel and 1 pixel of padding,  
you will get back the same size you start with.  If you use 5 by 5 with 3 pixels of padding,  
you'll end up with the same size you start  with. So generally odd numbered edge size   kernels are easier to deal with to make sure  you end up with the same thing you start with.
OK, so yeah, so as it says here, you've got an  odd numbered size ks by ks size kernel, then ks  
truncate divide to (ks//2). That's what slash  slash means will give you the right size. And so  
another trick you can do is you don't always have  to just move your window across by one each time.  
You could move it by a different amount each time.  The amount you move it by is called the stride.  
So, for example, here's a case of doing a stride  2, so a stride 2 padding 1. So we start out here  
and then we jump across 2 and then we jump across  2 and then we go to the next row. So that's called  
a stride 2 convolution. Stride 2 convolutions  are handy because they actually reduce the  
dimensionality of your input by a factor of 2.  And that's actually what we want to do a lot.
For example, with an autoencoder, we want to  do that. And in fact, for most classification  
architectures, we do exactly that. We keep  on reducing the, kind of the grid size by a  
factor of 2 again and again and again using  stride 2 convolutions with padding of 1.  
So that's strides and padding. So let's go ahead  and create a ConvNet using these approaches.  
Creating the ConvNet
So we're going to put, get our size of our  training set. This is all the same as before,  
number of categories, number of  digits, size of our hidden layer.
So. Previously, with our sequential linear  models, with our MLPs, we basically went  
from the number of pixels to the number  of hidden and then a ReLU and then the  
number of hidden to the number of outputs.  So here's the equivalent. With a convolution.
Now, the problem is that you can't just  do that because the output is not now 10  
probabilities for each item in our batch,  but it's 10 probabilities for each item in  
our batch for each of 28 by 28 pixels because  we don't even have a stride or anything. So you  
can't just use the same simple approach that we  had for MLP. We have to be a bit more careful.  
So to make life easier, let's create a little conv  function that does a Conv2d with a stride of 2  
optionally followed by an activation. So if act  is true, we will add in a ReLU activation. So this  
is going to either return a Conv2d or a little  Sequential containing a Conv2d followed by a ReLU.
And so now we can create a CNN  from scratch as a sequential model.  
And so since activation is True by default, this  is going to take our 28 by 28 image, starting with  
1 channel and creating an output of 4 channels.  So this is the number of in, this is the number  
of filters. Sometimes we'll say filters to  describe the number of, kind of, channels   that our convolution has, that's the number of  outputs. And it's very similar to the idea of  
the number of outputs in a linear layer, except  this is the number of outputs in your convolution.
So what I like to do when I create stuff like  this is I add a little comment just to remind   myself what is my grid size after this. So I had  a 28 by 28 input. So then I've then put it through  
a stride-2 conv. So the output of this will be  14 by 14. So then we'll do the same thing again,  
but this time we'll go from a 4 channel input  to an 8 channel output and then from 8 to 16.  
So by this point, we're now down to  a 4 by 4. And then down to a 2 by 2,  
and then finally we're down to a 1 by 1. So on  the very last layer, we won't add an activation.  
And the very last layer is going to create 10  outputs. And since we're now down to a 1 by 1, we  
can just call Flatten() and that's going to remove  those unnecessary unit axes. So if we take that,  
pop mini batch through it, we end up with  exactly what we want, 16 by 10. So for each  
of our 16 images, we've got 10 probabilities of  each possible digit. So if we take our training  
set and make it into 28 by 28 images, and  we do the same thing for a validation set,  
and then we create two datasets, one for each,  which are called train dataset and valid dataset.
And we're now going to train this on the GPU. Now,  if you've got a Mac, you can use a device called,  
well, if you've got an Apple Silicon Mac, you've  got a device called MPS, which is going to use  
your Mac's GPU. Or if you've got Nvidia, you can  use CUDA, which will use your Nvidia GPU. CUDA's  
10 times or more, possibly much more faster than  a Mac. So you definitely want to use Nvidia if  
you can. But if you're just running it on a  Mac laptop or whatever, you can use it MPS.
So basically, you want to know what device to use.  Do we want to use CUDA or MPS? You can check. If  
you can check torch.backends.mps.is_available()  to see if you're running on a Mac with MPS.  
You can check torch.cuda.is_available() to see  if you've got an Nvidia GPU, in which case you've  
got CUDA. And if you've got neither, of course,  you'll have to use the CPU to do computation.  
So I've created a little function here to_device,   which takes a tensor or a dictionary  or a list of tensors or whatever,  
and a device to move it to, and it just goes  through and moves everything onto that device.
Or if it's a dictionary, a dictionary of  things, values moved onto that device.  
So there's a handy little  function. And so we can create a  
custom collate function, which calls the  PyTorch default collation function and  
then puts those tensors onto our device.  And so with that, we've now got enough  
to train this neural net on the GPU. We created  this get_dls function in the last lesson.
So we're going to use that, passing in the  datasets that we just created and our default  
collation function. We're going to create  our optimizer using our CNNs parameters.  
And then we call fit(). Now fit(), remember, we also created  in our last lesson and it's done.  
So then what I did then was I reduced  the learning rate by a factor of four  
and ran it again. And eventually, yeah, I got to a  fairly similar accuracy to what we did on our MLP.  
So, yeah, we've got a convolutional network  working. I think that's pretty encouraging.   And it's nice that to train it, we didn't have  to write much code, right? We were able to  
use code that we had already built. We were  able to use the Dataset class that we made,   the get_dls function that we made,  and the fit function that we made.
And, you know, because those things are  written in a fairly general way, they work  
just as well for a ConvNet as they did for an  MLP. Nothing had to change. So that was nice.
Notice I had to take the model and put it on  the device as well. So that will go through and  
basically put all of the tensors that are in that  model onto the MPS or CUDA device, if appropriate.  
So if we've got a batch size of 64,  and as we do, 1 channel, 28 by 28,  
so then our axes are batch, channel, height,  width. So normally this is referred to as NCHW.
Convolution Arithmetic. NCHW and NHWC
So N, generally when you see N in  a paper or whatever, in this way,  
it's referring to the batch size. N  being the number, that's the mnemonic,   the number of items in the batch. C is the  number of channels, height by width, NCHW.  
TensorFlow doesn't use that. TensorFlow uses  NHWC. So we generally call that channels-last,  
since channels are at the end. And this one we  normally call channels-first. Now, of course,  
it's not actually channels-first. It's actually  channels-second, but we ignore the batch bit.  
In some models, particularly some more modern  models, it turns out the channels-last is faster.  
So PyTorch has recently added  support for channels-last. And so   you'll see that being used more and more as well.
All right, so a couple of comments and questions  from our chat. The first is Sam Watkins pointing  
Parameters in MLP vs CNN
out that we've actually had a bit of a win here,  which is that the number of parameters in our CNN  
is pretty small by comparison. So in the MLP  version, the number of parameters is equal to  
basically the size of this matrix, so m times nh.  
Oh, plus the number in this,  which will be nh times 10.  
And something that at some point we probably  should do is actually create something that  
allows us to automatically  calculate the number of parameters.  
And I'm ignoring the bias there, of course.
Let's see, what would be a good way to do that?  
Maybe np.product(). There we go. So  
what we could do is just  calculate this automatically  
by doing a little list comprehension here.  So there's the number of parameters across  
all of the different layers, so both bias  and weights. And then we could, I guess,  
just, well, we could just use, well, let's use  PyTorch. So we could turn that into a tensor  
and sum it up. Oops. So that's the number in  our MLP. And then the number in our simple CNN.  
So that's pretty cool. We've gone down from 40,000  to 5,000 and got about the same number there. Oh,  
thank you, Jonathan. Jonathan's reminding me that  there's a better way than np.product(o.shape),  
which is just to say o dot  number of elements: o.numel().  
Same thing. Very nice. Now, one person asked a very  good question, which is,  
CNNs and image size
I thought Convolutional Neural Networks can  handle any sized image. And actually, no,  
this convolutional network cannot handle any  sized image. This Convolutional Neural Network   only handles images that, once they go through  these stride-2 convs, end up with a 1 by 1.  
Because otherwise, you can't dot flatten it  and end up with 16 by 10. So we will learn how  
to create convnets that can handle any sized  input. But there's nothing particularly about  
a convnet that necessitates that it has  to be any sized input that it can handle.
Receptive fields
OK, so just let's briefly finish this section off  by talking about this, particularly I want to talk  
about the idea of receptive field. Consider this  1 input channel, 4 output channel, 3 by 3 kernel.  
So that's just to show you what we're doing here.  conv1, well, actually, so simple_cnn. simple_cnn.
This is the model we created. Remember, it was  like a Sequential model containing Sequential   models, because that's how our conv function  worked. So simple_cnn[0] is our first layer. It  
contains both a Conv and a ReLU. So simple_cnn[0,  0] is the actual Conv. So if we grab that,  
call it conv1, it's a 4 by 1 by 3 by 3. So,  number of outputs, number of input channels,  
and height by width of the kernel. And then  it's got its bias as well. So that's how we  
could deconstruct what's going on with our weight  matrices or our parameters inside a convolution.
Now, I'm going to switch over to Excel. So in  the lesson notes on the course website or on  
the forum, you'll find we've got an Excel.  You'll see we've got an Excel workbook.
Oh, Wasim reminded me that there is  a nice trick we can do. I do want   to do that actually because I love this trick.  
Oh, I just deleted everything though. Let's put  it all back. Here we go. Which is you actually  
don't need square brackets. The square brackets is  a list comprehension. Without the square brackets,   it's called a generator. And it, oh, no, you can't  use it there. Maybe that only works with NumPy.  
Ah, OK. So that's the list.  
No, that doesn't work either. So much for that.  I'm kind of curious now. Maybe torch.sum. No,  
just sum. Oh, OK. Well, I don't want to  use Python sum. That's interesting. I feel  
like all of them should handle  generators, but there you go.
OK. So open up the conv-example spreadsheet. And  what you'll see on the conv-example worksheet page  
Convolutions in Excel: conv-example.xlsx
is something that looks a lot like the number  7. And indeed, this is the number 7 that I got  
straight from MNIST. Let's see. OK. So you can  see over here, we have a number 7. This is a  
number 7 from MNIST that I have copied into  Excel. And then you can see over here, we've  
got the top edge kernel being applied. And over  here, we've got a right edge kernel being applied.  
This might be surprising you because you  might be thinking, where did it take Jeremy?   Microsoft Excel doesn't do Convolutional  Neural Networks. Well, actually, it does.
So if I zoom in in Excel, you'll see, actually,  these numbers are, in fact, conditional formatting  
applied to a bunch of spreadsheet cells. And so  what I did was I copied the actual pixel values  
into Excel and then applied conditional  formatting. And so now you can see what   the digit is actually made of. So you can see here  
I've created our top edge filter. And here  I've created our left edge filter. And so here  
I am applying that filter to that window. And  so here you can see it looks a lot like NumPy.  
It's just a sum product. And you might not be  aware of this, but in Excel, you can actually do  
broadcasting. You have to hit Apple  Shift Enter or Control Shift Enter,  
and it puts these little curly brackets  around it. It's called an array formula.   It basically lets you do broadcasting or  simple broadcasting in Excel. And so here's  
how you could say… this is how I created  this top edge filtered version in Excel. And the left edge version is exactly the same,  just a different kernel. And as you can see, if I  
click on it, it's applying this filter  to this input area and so forth.  
OK, so we can then… I just  arbitrarily pick some different  
values here. And so something to notice now  in my second layer, so here's Conv1, is Conv2,  
it's got a bit more work to do. We actually  need two filters because we need to add together  
this bit here applied to this with  this kernel applied and this bit here  
with this kernel applied. So you actually  need one set of 3 by 3 for each input.
And also, I want two separate outputs, so I  actually end up needing a 2 by 2 by 3 by 3  
weights matrix or weights tensor, I should say,  which you might remember is exactly what we had  
in PyTorch. We had a rank 4 tensor. So if I have a  look at this one, you see exactly the same thing.
This input is using this kernel applied  to here and this kernel applied to here.  
So that's important to remember that you  have these rank 4 tensors. And so then rather  
than doing stride-2 conv, I did something else  which is actually a bit out of favor nowadays,  
but it's another option, which is to do something  called max-pooling to reduce my dimensionality.
So you can see here I've got 28 by 28.  I've reduced it down here to 14 by 14.  
And the way I did it was simply to take  the max of each little 2 by 2 area.  
So that's all that's been done  there. So that's called max-pooling.
And so max-pooling has the same effect as a  stride-2 conv, not mathematically identical,  
the same effect, which is it does a convolution  and reduces the grid size by 2 on each dimension.  
So then how do we create a single  output if we don't keep doing this   until we get to 1 by 1, which  I'm too lazy to do in Excel?  
Well, one approach, and again, this is a little  bit out of favor as well, but one approach we   can do is we can take every one of these, we've  now got 14 by 14, and apply a dense layer to it.
And so what I've done here is I've got a  big, imagine this is basically all being  
flattened out into a vector. And so here  we've got some product of this by this,  
plus the sum product of this by this,  and that gives us a single number.  
And so that is how we could then optimize  that in order to optimize our weight matrices.
Now, and then, you know, the more modern approach,   we don't use this kind of dense layer much  anymore. It still appears a bit. The main  
place that you see this used is in a network  called VGG, which is very old now. I thought it  
might be 2013 or something, but it's actually  still used. And that's because for certain  
things like something called style transfer or in  general perceptual losses, people still find VGG  
seems to work better. So you still actually  see this approach nowadays sometimes.
The more common approach, however, nowadays is  we take the penultimate layer and we just simply  
take the average of all of the activations. So  the, you know, the nowadays we would simply,  
the Excel way of doing it would be literally  simply say AVERAGE of the penultimate layer.
And that is called global average pooling.  Everything has to has a fancy word, a fancy  
phrase, but that's all it is. Take the average  is called global average pooling, or you could  
take the max, whatever that  would be global max pooling. So anyway, the main reason I wanted to show you  this was to do something which I think is pretty  
interesting, which is to take something in our,  I'm just going to zoom out a little bit here.  
Let's take something in our  
max pool here. And I'm going to say trace  precedence to show you here it is the area that  
it's coming from. Okay. So it's coming from these  four numbers. Now for trace precedence again,   saying what's actually impacting this.  Obviously the kernels impacting it. And  
then you can see that the input area here is a  bit bigger. And then if I trace precedence again,  
then you can see the input area is bigger still.  So this number here is calculated from all of  
these numbers in the input. This area in the  input is called the receptive field of this unit.
And so the receptive field in this case  is 1, 2, 3, 4, 5, 6 by 6, right? And that  
means that a pixel way up here in the top,  right? Has literally no ability to impact  
that activation. It's not  part of its receptive field.   If you have a whole bunch of stride 2 convs, each  time you have one, the receptive field is going  
to get twice as big. So the receptive field at the  end of a deep network is actually very large. But  
the inputs closest to the middle of the receptive  field have the biggest kind of say in the output  
because they implicitly appear the most often in  all of these kind of dot products that are inside  
this convolutional window. So the receptive field  is not just like a single binary on off thing.
Certainly all the stuff that's not got precedence  here is not part of it at all. But the closer  
to the center of the receptive field, the  more impact it's going to have, the more   ability it's got to change this number. So the  receptive field is a really important concept  
and yeah, fiddling, playing around with Excel's  precedent arrows, I think is a nice way to  
see that, at least in my opinion.  And apart from anything else,  
it's great fun creating a Convolutional  Neural Network in Excel. I thought so anyway.
Okay, so let's take a seven minute break.   I'll see you back after that to talk about  a convolutional autoencoder. All right.
Okay, welcome back. We're going to have  a look now at the autoencoder notebook.  
Autoencoders
So we're just going to import all of our  usual stuff and we've got one more of our  
own modules to import now as well. And this  time we are going to switch to a different,  
we're going to switch to a different  dataset, which is the fashion MNIST dataset.  
We can take advantage of the  stuff that we did in 05_datasets  
and the Hugging Face stuff to load it.  So we've seen this a little bit before.
Back in our datasets one here.  
And we never actually built any models  with it. So let's first of all do that.  
So this is just, I'm going to convert  each thing, each image into a tensor,  
and that's going to be an in-place transform.  Remember we created this decorator.   And so we can call dataset dictionary with  the same name and so we can call dataset  
dictionary with transform. This  is all stuff we've done before.  
And so here we have our example of a sneaker.
All right. And we will create our collation  function, collating a dictionary for that dataset.  
That's something you should remind yourself. We  built that ourselves in the datasets notebook.  
And let's actually make our collate function   something that does to_device(),  which we wrote in our last notebook.  
And we'll create a little data_loaders function  here, which is going to go through each item in  
the dataset dictionary and create a DataLoader for  it and give us a dictionary of data loaders. Okay.
So,   okay. So now we've got a data loader for training  and a data loader for validation. So we can grab  
the x and y batch by just calling next()  on that iterator as we've done before.  
We can grab the, let's look at each  of these in turn actually. We've done  
all this before, but it's a couple of weeks ago. So just to remind you, we can  get the names of the features.  
And so we can then get, create an itemgetter()  for our y's. And we can, so we'll call that the  
label getter. We can apply that to our labels to  get the titles of everything in our mini batch.  
And we can then call our  show_images() that we created   with that mini batch, with those titles. And  here we have our fashion MNIST mini batch.
Okay. So let's create a classifier and  we're just gonna use exactly the same   code, copy and pasted from the previous  notebook. So here is our sequential model.  
And we are going to grab the parameters of the CNN  
and the CNN I've actually  moved it over to the device.  
The default device was what we created in our  last notebook. And as you can see, it's fitting.  
Speeding up fitting and improving accuracy
Now our first problem is it's fitting  very slowly, which is kind of annoying.
So why is it running pretty slowly? Let's think  about, let's have a look at our dataset. So  
when it's finally finished, let's take a  look at an item from the dataset. Actually,  
let's not look at the dataset. Let's actually  go all the way back to the dataset dictionary so  
before it gets transformed dataset dictionary  and let's grab the training part of that  
and let's grab one item. And  actually we can see here the problem.
For MNIST, we had all of the data loaded into  memory into a single big tensor, but this  
Hugging Face one is created in a much more kind  of normal way, which is, each image is a totally  
separate PNG image. It's not all pre-converted  into a single thing. Why is that a problem?
Well, the reason it's a problem is that  our DataLoader is spending all of its time  
decoding these PNGs. So if I train here,  
okay, so while I'm training, I can type htop  and you can see that basically my CPU is 100%  
used. Now that's a bit weird because I've  actually got 64 CPUs. Why is it using just  
one of them is the first problem. But why  does it matter that it's using 100% CPU? Well, the reason it matters,  let's run it again so you can see,  
why does it matter that our CPU is 100%  and why is it making it so slow? Well,  
the reason why is if we look at  nvidia-smi dmon that will monitor our  
GPUs utilization. I've got three GPUs, so  I say to choose just the zeroth index one.  
And you'll see this column here, sm, this stands  for symmetric multiprocessor. It's like the,  
it's the equivalent of like CPU usage. And  generally we're only using up 1% of our one GPU.
So no wonder it's so slow. So the first thing  we wanna do then is try to make things faster.  
Now, to make things faster, we wanna be  using more than one CPU to decode our PNGs.  
And as it turns out, that's actually  pretty easy to do. You just have to add a  
extra argument to your data loaders,  
which is here, num_workers. And so I  can say use eight CPUs, for example.  
Now, if I create, I recreate the data loaders  and then try to create, get the next one.
Oh, now I've got an error. And the error is  rather quirky. And what it's saying is, oh,  
you're now trying to use multiple processes. And  generally in Python and PyTorch, using multiple  
processes, things start to get complicated.  And one of the things that absolutely just   doesn't work is you can't actually have your  DataLoader put things onto the GPU in your  
separate processes. It just doesn't work. So the  reason for this error is actually because of the  
fact that we used a collate function that put  things on the device. That's incompatible,  
unfortunately, with using multiple  workers. So that's a problem.
And the answer to that problem, sadly,  
is that we would have to actually rewrite our  fit function entirely. So there's annoying thing  
number one, and we don't want to be rewriting  our fit function again and again. We want to   have a single fit function. So, okay. So there's  a problem that we're gonna have to think about.
Problem number two is that this is not very  accurate, 87%. Well, I mean, is it accurate?  
It's easy enough to find out. There's a  really nice website called paperswithcode.  
And it will tell you  
A little leaderboard. And we  can see whether we're any good.   And the answer is we're not very good at  all. So these papers had 96%, 94%, 92%.  
So yeah, we're not looking  great. So how do we improve that?
There's a lot of things we could try, but  pretty much all of them are going to involve  
modifying our fit function again and in reasonably  
complicated ways. So we still  got a bit of an issue there. Let's put that aside because what we actually  wanted to do is create an auto-encoder.  
Reminding what an auto-encoder is
So to remind you about what an auto-encoder is,  and we're gonna be able to go into a bit more  
detail now, we're gonna start with our input  image, which is gonna be 28 by 28. So it's the  
number 3, right? And it's a 28 by 28. And we're  gonna put it through, for example, a stride-2  
conv, stride-2. And that's going  to have an output of a 14 by 14.  
And we can have more channels. So say maybe  4, so this is 28 by 28 by 1. Let's do 14 by 14  
by 2. So we've reduced the height and width  by 2, but added an extra channel. So overall,  
this is a 2x decrease in parameters. And  then we could do another stride-2 conv,  
and that would give us a 7 by 7. And again,  we can choose however many channels we want,   but let's say we choose 4. So now compared to our  original, we've now got a times 4 reduction. And  
so we could do that a few times, or we could  just stay there. And so this is compressing.
And so then what we could do is then somehow  have a convolution layer or group of layers,  
which does a convolution  and also increases the size.  
There is actually something  called a transposed convolution,  
which I'll leave you to look up if you're  interested, which can do that. Also known  
as a rather weirdly, a stride-1/2 convolution. But  there's actually a really simple way to do this,  
which is to say, let's say you've got a bunch of  pixels is out. Let's say we've got a 3 by 3 pixels  
that looks like this, 1, 0, 1, 1,  say. We could make that into a 6 by 6  
very easily, which is we could simply,  
let's get these out. We could simply  copy that pixel there into the first 4,  
copy that pixel there into these 4. And so you can  see, and then copy this pixel here into these 4.  
And so we're simply turning each pixel into 4  pixels. And so this is called nearest neighbor  
upsampling. Now that's not a convolution,  that's just copying. But what we could  
then do is we could then apply a stride-1  convolution to that. Right? And that would  
allow us to double the grid size with the  convolution. And that's what we're gonna do.  
So our autoencoder is gonna need a deconvolutional  layer, and that's gonna contain two layers,  
up sampling nearest neighbor, scale factor of  2, followed by a conv2d with a stride of one.
Okay. And you can see for padding, I  just put kernel size // 2. So that's   a truncating division, cause that always  works for any odd sized kernel. As before,  
we will have an optional activation function, and  then we will create a sequential using *layers. So  
that's gonna pass in each layer as a separate  argument, which is what Sequential() expects.
Okay.  
So let's write a new fitness function. It goes  through, I just basically copied it over from  
our previous one, going through each epoch, but  I've pulled out eval into a separate function,  
but it's basically doing the same thing.   Okay. So here is our auto  encoder. And so we're going to,  
it's a bit tricky because I wanted to go  down by 1, 2, 3, to get to a 4 by 4 by 8,  
but starting at 28 by 28, you can't divide  that three times and get an integer.
So what I first do is I zero pad. So add padding  of 2 on each side to get a 32 by 32 input. So if  
I then do a conv with 2 channel output, that  gives us 16 by 16 by 2, and then again to get  
an 8 by 8 by 4, and then again to get a 4 by 4 by  8. So this is doing an 8x compression, and then we  
can call deconv() to do exactly the same thing  in reverse. The final one with no activation,  
and then we can truncate off those two pixels  off the edge, slightly surprisingly PyTorch,  
let's your pass -2 to zero padding to crop off the  final 2 pixels. And then we'll add a Sigmoid(),  
which will force everything to go between  0 and 1, which of course is what we need.
And then we will use mse_loss to compare  those pixels to our input pixels.  
And so a big difference we've got here now  is that our loss function is being applied  
to the output of the model and itself,  right? We don't have yb here, we have xb.  
So we're trying to recreate our original. And  again, this is a bit annoying that we have to   create our own fit function. Anyway, so we can  now see what is the mse_loss, and it's not,  
like, gonna be particularly human readable,  but it's a number we can see if it goes down.
And so then we can create,  
then we can do our SGD with the  parameters of our auto-encoder,   with mse_loss, call that fit  function, we just wrote, and  
I won't wait for it to run, cause as you can see,  it's really slow for reasons we've discussed.  
I've run it before. And what we want is  to see that the original, which is here,  
which is here, gets recreated.  And the answer is, oh, not really.  
I mean, they're roughly the same things,  but there's no point having an auto-encoder  
which can't even recreate the originals. The idea  would be that if these looked almost identical to  
these, then we'd say, wow, this is a fantastic  network at compressing things by eight times.  
So I found this like very fiddly to try and  get this to work at all. Something that I   discovered can get it to start training is  to start with a really low learning rate  
for a few epochs, and then increase the  learning rate after a few epochs. I mean,  
at least it gets it to train and show something  vaguely sensible, but let's see. Yeah, that still  
looks pretty crummy. This one here I got actually  by switching to Adam, and I actually removed  
the tricky bit. I removed these two as well. But yeah, I couldn't get this to like  
recreate anything very reasonable  or any reasonable amount of time. And why is this not working very well? There's so  many reasons it could be. Like do we need a better  
optimizer? Do we need a better architecture?  Do we need to use a Variational Auto-Encoder?  
There's a thousand things we could try,  but doing it like this is going to drive  
us crazy. We need to be able to really rapidly  try things and all kinds of different things.  
And so what I often see in projects or on  Kaggle or whatever, people's code looks kind  
of like this. It's all like manual. And then their  iteration speed is too slow. We need to be able to  
really rapidly try things. So we're not gonna keep  doing stuff manually anymore. This is where we  
take a halt and we say, okay, let's build  up a framework that we can use to rapidly  
try things and to understand when things  are working and when things aren't working.
Creating a Learner
So we're gonna start creating a learner.  
So what is a learner? It's basically the  idea is this learner is gonna be something   that we build, which will allow us to  try like anything that we can imagine  
very quickly. And we will build that on top  of that learner things that will allow us to   introspect what's going on inside our model,  will allow us to do multi-process CUDA to go  
fast. It will allow us to add things like  data augmentation. It will allow us to try   a wide variety of architectures quickly  and so forth. So that's gonna be the idea.
And of course we're gonna create it from scratch.  And so let's start with Fashion MNIST like before.  
And let's create a DataLoaders class,  
which is gonna look a bit like what we  had before, where we're just going to  
pass in, this is just, this couldn't be  simpler, right? We're just gonna pass in   two DataLoaders and store them away. And I'm gonna  create a @classmethod from dataset dictionary.  
And what that's gonna do is  it's gonna call DataLoader   on each of the dataset dictionary items with  our batch size and instantiate our class.
So if you haven't seen @classmethod before, it's  what allows us to say DataLoaders dot something in  
order to construct this. We could have put this in  __init__ just as well, but we'll be building more  
complex DataLoaders things later. So I thought we  might start by getting the basic structure right.
So this is all pretty much the same as  what we've had before. I'm not doing   anything on the device here, cause  as we know that didn't really work.
Okay. Oh, this is an old thing.  I don't need to_cuda() anymore.  
So we're gonna use to_device(),  which I think came from.  
There we go. So here's an example of  a very simple Learner that fits on  
one screen. And this is basically  gonna replace our fit function.   So a Learner is gonna be something that is  going to train or learn a particular model  
using a particular set of DataLoaders, a  particular loss function, some particular  
learning rate and some particular optimizer  or some particular optimization function.
Now, normally I, you know, most people  would often kind of store each of these   away separately by writing like  self.model equals model, blah,  
blah, blah, right? And as I think we've  talked about before, that's, you know,   that kind of huge amounts of boilerplate. It  just, it's more stuff that you can get wrong.  
And it's more stuff to mean that you have to  read to understand the code and yeah, don't like  
that kind of repetition. So instead we just call  fastcore.store_attr() to do that all in one line.
Okay, so that's the basic idea with  a class is to think about what's the   information it's gonna need. So you pass  that all to the constructor, store it away.  
And then our fit function is going to,  we've got the basic stuff that we have for  
keeping track of accuracy.   So this has only worked for stuff that's a  classification where we can use accuracy.  
Put the model on our device,   create the optimizer, store how many epochs we're  going through. Then for each epoch, we'll call the  
one epoch function and the one epoch function,  we're gonna either do train or evaluation. So we pass in True if we're training and False if  we're evaluating. And they're basically almost the  
same. We basically set the model to training  mode or not. We then decide whether to use  
the validation set or the training set based on  whether we're training. And then we go through  
each batch in the DataLoader and  call one batch. And one batch is  
then the thing which is going to  put our batch onto the device,  
call our model, call our loss function. And then,  if we're training, then do our backward step,  
our optimizer step and our zero gradient.  And then finally calculate our metrics or  
our stats. And so here's where we calculate our  metrics. So that's basically what we have there.
So let's go back to using an MLP. We call fit()  
and the way it goes.  
This is an error here, pointed out  by Kevin. Thank you. self.model.to().
One thing I guess we could try now is we think  that maybe we can use more than one process.  
So let's try that.  
Oh, it's so fast. I didn't even see.   There it goes. You can see all four CPUs  being used at once. Bang, it's done.
Okay, so that's pretty great. Let's see how fast  it looks here. Bump, bump. All right, lovely.
Okay, so that's a good sign. We've got a learner  that can fit things, but it's not very flexible.  
It's not gonna help us, for example, with our  autoencoder, because there's no way of like,  
just like, you know, changing which  things are used for predicting with,   or for calculating with. We can't use it for  anything except things that involve accuracy  
with a binary classification. Sorry… is that  right? Sorry, yeah, a multi-class classification.  
It's not flexible at all, but it's a  start. And so I wanted to basically   put this all on one screen so you can  see what the basic Learner looks like.
All right, so how do we do things  other than multi-class accuracy?
Metric class
I decided to create a Metric class. And basically  a Metric class is something where we are going  
to define subclasses of it that calculate  particular metrics. So for example, here,  
I've got a subclass of a Metric called Accuracy.  So if you haven't done subclasses before,  
you can basically think of this as saying,  please copy and paste all the code from here  
into here for me, but the bit that says def  calc(), replace it with this version. So in fact,  
this would be identical to copying and pasting  this whole thing, typing Accuracy here,  
and replacing the definition of calc() with  that. That's what is happening here when we  
do subclassing. So it's basically copying and  pasting all that code in there for us. It's  
actually more powerful than that. There's more  we can do with it, but in this case, this is all   that's happening with this subclassing. And this  is called, actually I'll leave that, that's fine.
Okay, so the Accuracy metric is here, and  then this is kind of our really basic Metric,  
which is we're gonna use for just for loss. And so  what happens is we're going to, let's for example,  
create an Accuracy metric object. We're  basically gonna add in mini batches of data,  
right? So for example, here's a mini batches of  inputs and predictions. Here's another mini batch  
of inputs and predictions. And then we're gonna  call .value and it will calculate the accuracy.
Now .value is a neat little thing. It  doesn't require parentheses after it  
because it's called a property. And  so a property is something that just   calculates automatically without having to  put parentheses. That's all a property is,  
well, property getter anyway. And so they look  like this, you give it a name. And so we are  
going to be, each time we call add(), we are  gonna be storing that input and that target.  
And also the number of items  in the mini batch optionally.  
For now, that's just always gonna be one.  And you can see here that we then call  
.calc(), which is gonna call the accuracy  calc(). So just see how often they're equal.  
And then we're going to  append to the list of values  
that calculation. And we're also gonna append  to the list of ns, in this case just one. And  
so then to calculate the value, we just do that.  So that's all that's happening for Accuracy.
And then we can do for loss, we can just use  Metric directly, cause Metric directly will just  
calculate the average of whatever it's passed.  So we can say, oh, add the number 0.6. So the  
target's optional. And we're saying this is a  mini batch of size 32. So it's gonna be the n.  
And then add the value 0.9 with a mini batch size  of 2, and then get the value. And as you can see,  
that's exactly the same as the weighted average  of 0.6 and 0.9 with weights of 32 and 2.
So we've created a Metric class. And so  that's something that we can use to create  
any metric we like just by overriding calc().   Or we could create totally things from scratch  as long as they have an add() and a value.
Okay, so we're now going to change our Learner.  And what we're gonna do is we're going to keep  
the same basic structure. So there's gonna be  fit(). It's gonna go through each epoch. It's  
gonna call one_epoch() passing in True and False  as for training and validation. one_epoch() is  
going to go through each batch in the DataLoader  and call one_batch(). one_batch() is going to do  
the prediction, get_loss(), and if it's training,  it's gonna do the backward() step and zero_grad().
But there's a few other things  going on. So let's take a look.  
Well, actually let's just look at  it in use first. So when we use it,   we're gonna be creating a Learner() with  the model, data loaders, loss function,  
learning rate, and some callbacks, which  we'll learn about in a moment. And we call   fit() and it's gonna do our thing. And  look, we're gonna have charts and stuff.
All right, so the basic idea is gonna  look very similar. So we're gonna call   fit(). So when we construct it, we're gonna be  passing in exactly the same things as before,  
but we've got one extra thing, callbacks, which  we'll see in a moment, store the attributes as   before and we're gonna be doing some stuff  with the callbacks. So when we call fit()  
for this number of epochs, we're gonna  store away how many epochs we're gonna do.   We're also gonna store away  the actual range that we're  
going to loop through as self.epoch. So  here's that looping through self.epoch.  
We're gonna create the optimizer using  the optimizer function and the parameters.  
And then we're gonna call _fit(). Now what on earth is _fit()? Why  didn't we just copy and paste?  
Decorator with callbacks
So this into here, why do this? It's because we've  created this special decorator with callbacks.
What does that do? So it's up here with  callbacks. With callbacks is a class.  
It's gonna just store one thing, which  is the name. In this case, the name is  
‘fit’. And what it's gonna do is… now this is the  decorator, right? So when we call it, remember,  
decorators get passed a function. So it's gonna  get passed this whole function and that's gonna  
be called f. So dunder call, remember is what  happens when a class is treated, an object is  
treated as if it's a function. So it's gonna get  passed this function. So this function is _fit.
And so what we wanna do is we wanna return a  different function. It's going to of course  
call the function that we were asked to call using  the arguments and keyword arguments we were asked  
to use. But before it calls that function, it's  going to call a special method called callback,  
passing in the string before, in this case,  before underscore fit. After it's completed,  
it's gonna call that method called callback  and passing the string after underscore fit.
And it's gonna wrap the whole  thing in a try except block.   And it's going to be looking for an exception  called CancelFitException. And if it gets one,  
it's not gonna complain. So let me explain  what's going on with all of those things.
Let's look at an example of a callback.  
So for example, here is a Callback called  DeviceCB, device callback. And before_fit() will  
be called automatically before that underscore fit  method is called. And it's going to put the model  
onto our device, CUDA or MPS, if we have one,  otherwise it'll just be on GPU. So what's gonna  
happen here? So it's going to call, we're gonna  call fit(). It's gonna go through these lines of  
code. It's then gonna call _fit(). _fit() is not  this function. _fit() is this function with f is  
this function. So it's going to call our  learner dot callback passing in before_fit.  
And callback() is defined here. What's  callback() gonna do? It's gonna be passed  
the string before_fit(). It's going to  then go through each of our callbacks  
sorted based on their order. And you can  see here, our callbacks can have an order  
and it's going to look at that callback and  try to get an attribute called before_fit.  
And it will find one. And so then it's going to  call that method. Now, if that method doesn't  
exist, it doesn't appear at all, then getatrr()  will return this instead. Identity is a function  
just here. This is an identity function. All  it does is whatever arguments it gets passed,  
it returns them. And if it's not  passed any arguments, it just returns.
Python recap
So there's a lot of Python going on here. And  that is why we did that foundations lesson.  
And so for people who haven't done a lot of  this Python, there's gonna be a lot of stuff to  
experiment with and learn about.  And so do ask on the forums, if  
any of these bits get confusing, but the  best way to learn about these things is to   open up this Jupyter notebook and try and  create really simple versions of things.
So for example, let's try identity().  identity(), how exactly does identity work?  
I can call it and it gets nothing. I can call it  with 1, it gets back 1. I could call it with ‘a’,  
gets back ‘a’, call it with ‘a’, 1.  
Call it with ‘a’, 1 and get ‘a’, 1.
And how is it doing that exactly? So remember  we can add a break point and this would be a  
great time to really test your debugging skills.  Okay, so remember in our debugger, we can hit h  
to find out what the commands are, but you  really should do a tutorial on the debugger   if you're not familiar with it. And then we can  step through each one. So I can now print args.
And there's actually a trick which I like is  that args is actually a command, funnily enough,  
which will just tell you the arguments to any  function, regardless of what they're called,   which is kind of nice. And so then  we can step through by pressing n  
and after this, we can check  like, okay, what is x now?  
And what is args now? Right?, so remember  to really experiment with these things.
So anyway, we're gonna talk about  this a lot more in the next lesson.  
But before that, if you're not  familiar with try-except blocks,  
you know, spend some time practicing them.  If you're not familiar with decorators,   well, we've seen them before. So go back  and look at them again really carefully.  
If you're not familiar with the debugger, practice  with that. If you haven't spent much time with  
getattr, remind yourself about that. So try to get yourself really  familiar and comfortable  
as much as possible with the pieces, because  if you're not comfortable with the pieces,   then the way we put the pieces together is  gonna be confusing. There's actually something  
in education in kind of the theory of education  called cognitive load theory. And the theory of  
cognitive, basically cognitive load theory  says, if you're trying to learn something,  
but your cognitive load is really high because  of all lots of other things going on at the same  
time, you're not gonna learn it. So it's gonna  be hard for you to learn this framework that  
we're building if you have too much cognitive  load of like what the hell's a decorator or   what the hell's getattr or what does sorted do  or what's partial, you know, all these things.
Now, I actually spent quite a bit of time  trying to make this as simple as possible,  
but also as flexible as it needs  to be for the rest of the course.  
And this is as simple as I could get  it. So these are kind of things that  
you actually do have to learn. But in doing  so, you're gonna be able to write some really  
powerful and general code yourself. So hopefully  you'll find this a really valuable and mind  
expanding exercise in bringing high level software  engineering skills to your data science work.
Okay, so with that, this looks like a good place   to leave it and look forward  to seeing you next time. Bye.

---

# 16

Hi there and welcome to Lesson 16 where we  are working on building our first flexible 
training framework: the learner. And I've got some very good news, which is  
that I have thought of a way of doing it a little bit more gradually and simply,   actually, than last time. So that should make things a bit easier. 
So we're going to take it a bit more step by step. So we're working in the 09_learner notebook today. 
And we've seen already this  Basic Callbacks Learner. 
And so the idea is that… we've seen so far  this Learner which wasn't flexible at all, 
but it had all the basic pieces,  which is we've got a fit method,  
we’re hardcoding that we can only calculate accuracy and average loss.  We're hardcoding, we're putting  things on a default device,  
hardcoding a single learning rate.  But the basic idea is here,  we go through each epoch  
and call one_epoch to train or evaluate depending on this flag. 
And then we loop through  each batch in the DataLoader.  And one_batch is going to grab the x and y  parts of the batch, call the model, call the 
loss function, and if we're  training, do the backward pass. 
And then print out, well, calculate  the statistics for our accuracy. 
And then at the end of an epoch, print that out. So it wasn't very flexible. 
But it did do something. So that's good.  So what we're going to do now is we're  going to do is an intermediate step. 
We're going to look at a, what I'm  calling a Basic Callbacks Learner.  And it actually has nearly all the  functionality of the full thing. 
After we look at this Basic Callbacks Learner,  we're then going to, after creating some callbacks 
and metrics, we're going to look at  something called the flexible learner.  So let's go step by step. So the Basic Callbacks Learner  
Basic Callback Learner
looks very similar to the previous Learner. It's got a fit function, which is going to go  
through each epoch, calling one_epoch with training on and then training off. 
And then one_epoch will go through each batch  and call one_batch and one_batch will call 
the model, the loss_func. And if we're training   it will do the backward step. So that's all pretty similar,  
but there's a few more things going on here. For example, if we have a look at fit,  
you'll see that after creating the optimizer, so we call self.opt_func, so opt_func here  
defaults to SGD. So we instantiate an SGD object passing in our  
model's parameters and the requested learning rate.  And then before we start looping through one  epoch at a time, now we've set epochs here, 
we first of all call self.callback  and passing in ‘before_fit’.  Now what does that do? So self.callback is here  
and it takes some method names,  in this case it's ‘before_fit’,  and it calls a function  called run callbacks ~run_cbs. 
It passes in a list of our callbacks and  the method name, in this case ‘before_fit’.  So run callbacks is something that's going  to go for each callback and it's going to 
sort them in order of their ‘order’ attribute.  And so there's a base class to our callbacks  which has an order of 0, so our callbacks 
are all going to have the same  order of 0 unless you ask otherwise.  So here's an example of a callback. So before we look at how callbacks work,  
let's just run a callback. So we can create a ridiculously  
simple callback called completion  callback ~CompletionCB, which before we  start fitting a new model, it  will set its count attribute to 0. 
And each batch it will increment that, and  after completing the fitting process it will 
print out how many batches we've done. So before we even train a model, we could   just run manually before_fit, after_batch, and after_fit using this run_cbs. 
And you can see it's ended up  saying “Completed 1 batches”.  So what did that do? So it went through each of the cbs in this  
list, there's only one, so it's going to look at the one cb, and it's going to try to use  
getattr to find an attribute with this name, which is ‘before_fit’. 
So if we try that manually, so this is the  kind of thing I want you to do if you find  anything difficult to understand  is do it all manually. 
So create a callback, set it to cbs[0],  just like you're doing in a loop, right? 
And then find out what happens  if we call this and pass in this,  
and you'll see it's returned a method.  And then what happens to that method? It gets called. 
So let's try calling it. There we are. 
So that's what happened when we  call the before_fit, which doesn't   do anything very interesting. But if we then call after_batch,  
and then we call after_fit, there it is, right? So yeah, make sure you don't just run code  
willy-nilly, but understand it by experimenting with it. 
And I don't always experiment  with it myself in these classes.  Often I'm leaving that to you, but sometimes  I'm trying to give you a sense of how I would 
experiment with code if I was learning it. So then having done that, I would   then go ahead and delete those cells. But you can see I'm using this interactive  
notebook environment to explore and learn and understand.  And so now we've got an end. If I haven't created a simple example of something  
to make it really easy to understand, you should do that, right?  Don't just use what I've already created  or what somebody else has already created. 
So we've now got something that  works totally independently.  We can see how it works. This is what a callback does.  So a callback is something  which will look at a class. 
A callback is a class where you can define one  or more of before, after_fit, before, after_batch 
and before, after_epoch. So it's going to go through and run all the  
callbacks that have a before_fit method before we start fitting. 
Then it'll go through each epoch and call  one_epoch with training and one epoch with  evaluation. And then when that's all done,  
it will call after_fit callbacks. And one_epoch will, before it  
starts on enumerating through the  batches, it will call before_epoch. 
And when it's done, it will call after_epoch. The other thing you'll notice is that there's a  
Exceptions
try-except immediately before every before method and immediately after every after  
method, there's a try and there's an except. And each one has a different thing to look for,  
CancelFitException, CancelEpochException, and CancelBatchException. 
So here's the bit which goes through each  batch, calls before_batch, processes the batch, 
calls after_batch. And if there's an exception that's of   type CancelBatchException, it gets ignored. So what's that for? 
So the reason we have this is that any of  our callbacks could raise any one of these 
three exceptions to say, I don't  want to do this batch, please. 
So maybe we'll look at an  example of that in a moment.  So we can now train with this. So let's call create a little get_model function  
Train with Callbacks
that creates a sequential model with just some linear layers.  And then we'll call fit. And it's not telling us anything interesting  
because the only callback we added was the completion callback. 
That's fine. It's training, it's doing something.  And we now have a trained model. I just didn't print out any metrics or anything  
because we don't have any callbacks for that. That's the basic idea.  So we could create a, maybe we could call  it a SingleBatchCB, which after_batch, 
after a single batch, it  raises a CancelFitException. 
So that's a pretty, I mean, I suppose that  could be kind of useful actually, if you want  to just run one batch model to make sure it works. So we could try that. 
So now we're going to add to  our list of callbacks, the  
single batch callback. Let's try it. 
And in fact, you know, we probably want this. Let's just have a think here. 
That's fine. Let's run it.  There we go. So it ran and nothing happened.  And the reason nothing happened is  because this canceled before this ran. 
So we could make this run second  by setting its order to be higher. 
And we could say just order=1  because the default order is 0. 
And we thought in order of the order attribute. Actually let's use CancelEpochException. 
There we go. That way it'll run the final fit. 
There we are. So it did one batch for the  
training and one batch for the evaluation. So that's a total of two batches.  So remember, callbacks are not  a special magic part of like  
the Python language or anything. It's just a name we use to refer   to these functions or classes  or callables, more accurately, 
that we pass into something that will then  call back to that callable at particular times. 
And I think these are kind of interesting  kinds of callbacks because these callbacks  have multiple methods in them. So is each method a callback? 
Is each class with all those methods a callback? I don't know.  I tend to think of the class with all  the methods in as a single callback. 
I'm not sure if we have  great nomenclature for this. 
All right. So let's actually   try to get this doing something more  interesting by not modifying the learner  at all, but just by adding callbacks, because  that's the great hope of callbacks, right? 
So it would be very nice if it  told us the accuracy and the loss. 
Metrics class: accuracy and loss
So to do that, it would be great to have  a class that can keep track of a metric. 
So I've created here a Metric class. And maybe before we  
look at it, we'll see how it works. You could create, for example, an accuracy   metric by defining the calculation necessary to calculate the accuracy metric, which is the  
mean of how often do the inputs equal the targets. 
And the idea is you could then  create an accuracy metric object.  You could add a batch of inputs and targets  and add another batch of inputs and targets 
and get the value. And there you would get the 0.45 accuracy. 
Or another way you could do it would be just to  create a metric which simply gets the weighted 
average, for example, of your loss. So you could add 0.6 as the loss with a batch   size of 32, 0.9 as a loss and a batch size of 2. 
And then that's going to give us a weighted  average loss of 0.62, which is equal to this 
weighted average calculation. So that's like one way we could kind of   make it easy to calculate metrics. So here's the class. 
Basically we're going to keep track of all  of the actual values that we're averaging 
and the number in each mini batch. And so when you add a mini batch, we call  
calculate, which for example, for Accuracy, remember this is going to override  
the parent classes calculate. So it does the calculation here.  And then we'll add that to our list of values. We will add to our list of batch sizes,  
the current batch size. And then when you calculate the value,  
we will calculate the weighted sum. Sorry, the weighted mean, weighted average. 
Now notice that here value, I didn't  have to put parentheses after it.  And that's because it's a property. I think we've seen this before. 
So just to remind you, property just means  you don't have to put parentheses after it  to get the calculation to happen. All right. 
So just let me know if anybody's got  any questions up to here, of course. 
So we now need some way to use this metric  in a callback to actually print out. 
Device Callback
The first thing I'm going to do though, is I'm  going to create one more, one useful metric  first, a very simple one, just two lines  of code called the device callback. 
And that is something which is going to allow  us to use CUDA or for the Apple GPU or whatever 
without the complications we had before of,  you know, how do we have multiple processes 
in our DataLoader and also use our  device and not have everything fall over.  So the way we could do it is we could say  before fit, put the model onto the default 
device and before each batch is run, put that  batch onto the device, because look what happened 
in the, this is really, really  important. In the Learner  absolutely everything is put inside self  dot, which means it's all modifiable. 
So we go for self dot iteration number, comma  self dot the batch itself and numerating the 
DataLoader.  And then we call one_batch, but before it,  we call the callback so we can modify this. 
Now how does the callback  get access to the Learner?  Well what actually happens is we go through  each of our callbacks and put, set an attribute 
called learn equal to the Learner. And so that means in the callback itself,  
we can say self.learn.model. And actually we could make this  
a bit better, I think. So make it like, maybe   you don't want to use a default device. So this is where I would be inclined to  
add a constructor and set device and we could default it to the default device, of course. 
And then we could use that instead and  that would give us a bit more flexibility. 
So if you wanted to train on some  different device, then you could. 
I think that might be a slight improvement.  Okay. So there's a callback we can use to put  
things on CUDA and we could check that it works by just quickly going back to our old Learner here,  
remove the SingleBatchCB() and replace it with DeviceCB(). 
Yep still works. So that's a good sign. 
Metrics Callback
Okay so now let's do our Metrics. Now   of course we couldn't use metrics  until we built them by hand. 
The good news is we don't have to write every  single metric now by hand because they already  exist in a fairly new project called torcheval,  which is an official PyTorch project. 
And so torcheval is something that gives  us… actually, I came across it after I had 
created my own metric class. But it actually looks pretty similar   to the one that I built earlier. So you can install it with pip. 
I'm not sure if it's on Conda yet, but it probably  will be soon by the time you see the video. 
I think it's pure Python anyway so  it doesn't matter how you install it.  And yeah it has a pretty similar approach  where you call .update and you call .compute 
so they're slightly different names but they're  basically super similar to the thing that  we just built. But there's a nice good list of  
metrics to pick from.  So because we've already built our own now  that means we're allowed to use theirs. 
So we can import the MultiClassAccuracy  metric and the Mean metric. 
And just to show you they look very, very similar. If we call MultiClassAccuracy and we can pass  
in a mini-batch of inputs and targets and compute and that all works nicely. 
Now these, in fact, it's exactly  the same as what I wrote.  We both added this thing called  reset which basically resets it. 
And so obviously we're going to be wanting to  do that probably at the start of each epoch. 
And so if you reset it and then try to compute  you'll get NaN because you can't get accuracy. 
Accuracy is meaningless when  you don't have any data yet.  So let's create a metrics callback  «MetricsCB» so we can print out our metrics. 
I've got some ideas to improve this which  maybe I do this week but here's a basic working  version slightly hacky but it's not too bad. 
So generally speaking one thing I noticed  actually is I don't know if this is considered  a bug but a lot of the metrics didn't seem  to work correctly in torcheval when I had 
tensors that were on the  GPU and had requires grad. 
So I created a little to_cpu function which  I think is very useful and that's just going 
to detach the… so detach takes the tensor and  removes all the gradient history, the computation 
history used to calculate a  gradient and puts it on the CPU.  It will do the same for dictionaries of  tensors, lists of tensors and tuples of tensors. 
So our metrics callback basically  here's how we're going to use it.  So let's run it. So here we're creating a metrics callback object  
and saying we want to create a metric called accuracy.  That's what's going to print out. 
And this is the metrics object we're  going to use to calculate accuracy.  And so then we just pass that  in as one of our callbacks. 
And so you can see what it's going to do is it's  going to print out the epoch number, whether 
it's training or evaluating, so training set  or validation set, and it'll print out our 
metrics and our current status. Actually we can simplify that. 
We don't need to print those  bits because it's all in the   dictionary now. Let's do that. 
There we go. So let's take a look at how this works. 
So we are going to be creating with, for the  callback we're going to be passing in the 
names and object, metric objects,  for the metrics to track and print. 
So here it is here, **metrics. So we've seen ** before. 
And as a little shortcut, I decided that it  might be nice if you didn't want to write  accuracy equals, you could  just remove that and run it. 
And if you do that, then it will give it a  name and it'll just use the same name as the  class. And so that's why you can either pass in. 
So *ms will be a tuple. Well, ms it's going to be pulled out.  So it's just passing a list of positional  arguments, which we turned into a tuple, or 
you can pass in named arguments that  will be turned into a dictionary.  If you pass in positional arguments, then  I'm going to turn them into named arguments 
in the dictionary by just  grabbing the name from their type.  So that's where this comes from. That's all that's going on here. 
Just a little shortcut, a bit of convenience. So we store that away. 
And this is, yeah, this is the bit I think  I can simplify a little bit, but I'm just  adding manually an additional metric,  which is I'm going to call the loss. 
And that's just going to be the  weighted average of the losses. 
So before we start fitting, we're going to  actually tell the Learner that we are the 
Metrics callback.  And so you'll see later where  we're going to actually use this. 
Before each epoch, we will  reset all of our metrics. 
After each epoch, we will  create a dictionary of the  
keys and values, which are the actual strings that we want to print out.  And we will call _log, which  for now we'll just print them. 
And then after each batch, this is  the key thing, we're going to actually   grab the input and target. 
We're going to put them on the CPU. And then we're going to  
go through each of our metrics and call .update. So remember the update in the metric  
is the thing that actually says here's a batch of data, right?  So we're passing in the batch of data,  which is the predictions and the targets. 
And then we'll do the same thing  for our special loss metric,   passing in the actual loss and the size of our mini-batch. 
And so that's how we're able  to get this, yeah, this actual  
running on the NVIDIA GPU and showing our metrics.  And obviously there's a lot of room to improve  how this is displayed, but all the information's 
we needed here, and it's just a  case of changing that function. 
Okay. So that's our kind   of like intermediate complexity learner. We can make it more sophisticated,  
Flexible Learner: @contextmanager
but it's still exactly, it's still going to fit in a single screen of code.  So this is kind of my goal here was to  keep everything in a single screen of code. 
This first bit is exactly the same as before,  but you'll see that the one_epoch and fit 
and batch has gone from,  let's see what it was before. 
It's gone from quite a lot of code, all this,  
to much less code. And the trick to doing   that is I decided to use a @contextmanager. We're going to learn more about context managers  
in the next notebook, but basically I originally last week I was saying I was going to do this with   a decorator, but I realized a context manager is better. 
Basically what we're going to do  is we're going to call our before   and after callbacks in a try-except block. 
And to say that we want to use the callbacks  and the try and except block, we're going  to use a with statement. So in Python, a with statement says everything  
in that block call our context manager before and after it. 
Now there's a few ways to do that, but one  really easy one is using this context manager  decorator and everything up until the  yield statement is called before your code. 
Where it says yield, it then calls your code  and then everything after the yield is called 
after your code. So in this case, it's going to be try,   self.callback, before_name, where name is fit. And then it will call for self.epoch etc. 
Because that's where the yield is. And then it'll call self.callback(after_fit),  
except. Okay and   now we need to grab the CancelFitException. So all of the variables that you have in Python,  
all live inside a special dictionary called globals().  So this dictionary contains all of your variables. So I can just look up in that dictionary,  
the variable called CancelFit –with a capital F– Exception. 
So this is except: CancelFitException. So this is exactly the same then as  
this code, except the nice  thing is now I only have to  write it once rather than at least three times. And I'm probably going to want more of them. 
So I tend to think it's worth refactoring a  code when you have duplicate code, particularly 
here. We had the same code three times.  So that's going to be more  of a maintenance headache.  We're probably going to want to  add callbacks to more things later. 
So by putting it into a context manager just once,  I think we're going to reduce our maintenance 
burden. That's what we do because I've had a similar   thing in fast.ai for some years now, and it's been quite convenient. 
So that's what this context manager is about. 
Yeah, other than that, the  code's exactly the same.  So we create our optimizer and then with our  callback context manager for fit, go through 
each epoch, call one_epoch, set it to training  or non-training mode based on the argument 
we pass in, grab the training or validation  set based on the argument we pass in, and 
then using the context manager for epoch,  go through each batch in the DataLoader, 
and then for each batch in the  DataLoader using the batch context. 
Now this is where something  gets quite interesting.  We call predict, get_loss, and if we're  training, backward, step and zero_grad. 
But previously we actually called  self.model, etc., self.loss_func, etc. 
So we go through each batch and  call before_batch, do the batch. 
Oh, sorry, that's our slow version. Wait, what are we doing?  Oh, yes, we're going to be over here. 
Okay, back where we are, yes.  So previously we were calling the model,  calling the loss_func, calling lost.backward, 
opt.step, opt.zero_grad, but now we are  
calling instead, self.predict,  self.get_loss, self.backward,  and how on earth is that working  –because they're not defined here at all. 
And so the reason I've decided to do  this is it gives us a lot of flexibility.  We can now actually create our own way of  doing predict, get_loss, backward, step, and 
zero_grad, in different situations, and  we're going to see some of those situations. 
So what happens if we call  self.predict and it doesn't exist?  Well, it doesn't necessarily cause an error. What actually happens is it calls a special magic  
method in Python called dunder getattr, as we've seen before.  And what I'm doing here is I'm saying, okay,  well, if it's one of these special five things, 
don't raise an AttributeError, which this  is the default thing it does, but instead… 
create a callback, or actually I should say  call self.callback, passing in that name. 
So it's actually going to  call self.callback(‘predict’). 
And self.callback is exactly the same as before. And so what that means now is to make this   work exactly the same as it did before, I need a callback which does these five things. 
And here it is. I'm going to call it train callback.  So here are the five things, predict,  get_loss, backwards, step, and zero_grad. 
Flexible Learner: Train Callback
So they're here, predict, get_loss,  backwards, step, and zero_grad. 
Okay, so they're almost exactly the same as  what they looked like in our intermediate 
Learner, except now I just need to have self.learn  in front of everything, because, we remember,  this is a Callback, it's not the Learner. And so for a Callback, the Callback  
can access the Learner using self.learn. So self.learn.preds, there's self.learn.model,   passing in self.learn.batch, and just the independent variables. 
Ditto for the loss, calls the loss function,   backward, step, zero_grad. So that's, at this point, this isn't  
doing anything that it wasn't doing before. But the nice thing is now if you want to use   HuggingFace Accelerate, or you want something that works on HuggingFace data styles  
dictionary things, or whatever,  you can actually change exactly 
how it behaves by creating  a Callback for training. 
And if you want everything except one thing  to be the same, you can inherit from TrainCB.  So this is, I've not tried this before,  I haven't seen this done anywhere else. 
So it's a bit of an experiment. So I'm interested to hear how you go with it. 
Flexible Learner: Progress Callback (fastprogress)
And then finally, I thought it'd  be nice to have a progress bar.  So let's create a progress callback. And the progress bar is going to show on it  
our current loss and going to create a plot of it.  So I'm going to use a project that we created  called fastprogress, mainly created by the 
wonderful Sylvain. And basically, fastprogress is  
a very nice way to create  very flexible progress bars.  So let me show you what it looks like first. So let's get the model and train. 
And as you can see, it actually in real  time updates the graph and everything.  There you go. That's pretty cool. 
So that's the progress bar, the metrics callback,  the device callback, and the training callback 
all in action. So before we fit,   we actually have to set self.learn.epochs. Now that might look a little bit weird,  
but self.learn.epochs is the thing that we loop through for self.epoch in. 
So we can change that. So it's not just a normal range,   but instead it is a progress bar around a range. We can then check, 
remember I told you that the Learner is  going to have the metrics attribute applied?  We can then say, oh, if the Learner has a  metrics attribute, then let's replace the 
_log method there with ours. And our one instead will   write to the progress bar. Now this is pretty simple. 
It looks very similar to before, but we could  easily replace this, for example, with something  that creates an HTML table, which is another  thing fastprogress does or other stuff like 
that. So you can see, we can modify.  The nice thing is we can modify  how our metrics are displayed. 
So that's a very powerful thing that Python  lets us do is actually replace one piece of  code with another. And that's the whole purpose of why the  
metrics callback had this _log separately. So why didn't I just say print here? 
That's because this way classes can  replace how the metrics are displayed. 
So we could change that to like send them  over to Weights and Biases, for example, or, 
you know, create visualizations or so forth. 
So before_epoch, we do a very similar thing. The self.learn.dl iterator.  We change it to have a  progress bar wrapped around it. 
And then after each bar, we set the  progress bar's comment to be the loss. 
It's going to show the loss on  the progress bar as it goes.  And if we've asked for a plot, then we  will append the losses to a list of losses. 
And we will update the graph  with the losses and the  
batch numbers. So there we have it.  We have a nice working Learner, which is, I  think, the most flexible Learner that training 
loop probably that's, I  hope, has ever been written.  Because I think the fastai2 one was the most  flexible that had ever been written before. 
And this is more flexible. And the nice thing is, you can make this your own. 
You know, you can, you know, fully  understand this training loop. 
So it's kind of like you can use a framework,  but it's a framework in which you're totally 
in control of it. And you can make it work exactly how you want to.  Ideally, not by changing the Learner  itself, ideally by creating callbacks. 
But if you want to, you could certainly, like,  look at that, the whole Learner fits on a  single screen. So you could certainly change that. 
We haven't added inference yet, although  that shouldn't be too much to add.  I guess we have to do that at some point. 
Okay, now, interestingly, I love  this about Python, it's so flexible. 
When we said self.predict, self.get_loss,  I said if they don't exist, then it's going 
to use getattr, and it's going to  try to find those in the callbacks. 
And in fact, you could have multiple callbacks  that define these things, and then they would  chain them together, which  would be kind of interesting. 
But there's another way we could make these  exist, which is that we could subclass this. 
So let's not use TrainCB, just  to show us how this would work. 
TrainingLearner subclass, adding momentum
Instead, we're going to use a subclass. So here I'm going to subclass Learner,  
and I'm going to override the five. Well, it's not exactly overriding,   I didn't have any definition of them before. So I'm going to define the five  
directly in the Learner subclass. So that way, it's never going to end up going   to getattr, because getattr is only called if something doesn't exist. 
So here, it's basically, all these five are  exactly the same as in our train callback, 
except we don't need self.learn anymore,  we can just use self because we're now   in the Learner.  But I've changed zero_grad  to do something a bit crazy. 
I'm not sure if this has been done before,  I haven't seen it, but maybe it's an old  trick that I just haven't come across. But it occurred to me, zero_grad,  
which remember is the thing  that we call after we take the  optimizer step, doesn't actually  have to zero the gradients at all. 
What if instead of zeroing the gradients,  we multiplied them by some number, like say 
0.85? Well, what would that do? 
Well, what it would do is it would mean that  your previous gradients would still be there, 
but they would be reduced a bit. And remember, what happens in PyTorch is PyTorch  
always adds the gradients to the existing gradients.  And that's why we normally have to call zero_grad. But if instead we multiply the gradients by some  
number, I mean, we should really make this a parameter.  Let's do that, shall we? So let's create a parameter. 
So probably there's a few ways we could do this. Well, let's do it properly. 
We've got a little bit of time. So we could say,  
well, maybe I'll just copy  and paste all those over here. 
And we'll add momentum equals 0.85. self.momentum equals momentum. 
And then super. So make sure you call the super classes,  
passing in all the stuff.  We could use delegates for this and kwargs. That would be possibly another way of doing it. 
But let's just do this for now. Okay.  And then so there we wouldn't make it 0.85. We would make it self.momentum. 
So you'll see now still trains, but there's  no TrainCB callback anymore in my list. 
I don't need one because I have defined  the five methods in the subclass. 
Now this training at the same learning  rate for the same time, the accuracy,  
let's run them all. 
Yeah, this is a lot like  gradient accumulation callback.  Kind of cooler, I think. 
Okay. So the let's see, the loss has gone from 0.8  
to 0.55 and the accuracy has gone from about 0.7 to about 0.8. 
So they've improved. Why is that?  Well, we're going to be learning a lot  more about this pretty shortly, but  
basically what's  happening here is we have just  implemented in a very interesting  way, which I haven't seen done  before, something called momentum. 
And basically what momentum does is it say,  imagine you've got some kind of complex contour 
loss surface. So imagine these are hills   with a marble, very similar. And your marble's up here. 
What would normally happen with gradient descent  is it would go in the direction downhill. 
Which is this way. So it would go over here and then over here.  Right? Very slow. 
What momentum does is the first step's the same.  And then the second step says, oh, I wanted  to go this way, but I'm going to add together 
the previous direction plus the new direction,  but reduce the previous direction a bit. 
So that would actually make me end up about here. And then the second one does the same thing. 
And so momentum basically makes you much  more quickly go to your destination. 
So normally momentum is done, the reason I  did it this way, partly to show you, is just 
a bit of fun, a bit of interest, but it's  very useful because normally momentum, you 
have to store a complete copy basically of  all the gradients, the momentum version of 
the gradients, so that you can kind of keep  track of that running exponentially weighted  moving average. But using this trick, you're actually using  
the dot grad themselves to store the exponentially weighted moving average. 
So anyway, there's a little bit of fun, which  hopefully, particularly those of you who are  interested in accelerated optimizers and  memory saving might find a bit inspiring. 
Learning Rate Finder Callback
All right. There's one more  
callback I'm going to show before the  break, which is the wonderful learning  rate finder.  I'm assuming that anybody who's watching this  already is familiar with the learning rate 
finder from fastai. If you're not, there's   lots of videos and tutorials around about it. It's an idea that comes from a paper by Leslie  
Smith from a few years ago. And the basic idea   is that we will increase the learning rate. I should have put titles on this. 
The X axis here is learning rate. The Y axis here is loss. 
We increase the learning rate gradually over  time and we plot the loss against the learning 
rate and we find how high can we bring the  learning rate up before the loss starts getting 
worse. And you   kind of want roughly where about the steepest  slope is, so probably here it'd be about 0.1. 
So it'd be nice to create a learning rate finder. So here's a learning rate finder callback. 
So what a learning rate finder needs to do,  well, you have to tell it how much to multiply  the learning rate by each batch. Let's say we add 30% to the  
learning rate each batch. So we'll store that.  So before we fit, we obviously need to keep  track of the learning rates and we need to 
keep track of the losses because those  are the things that we put on a plot. 
The other thing we have to do is  decide when do we stop training.  So when has it clearly gone off the rails? And I decided that if the loss is three times  
higher than the minimum loss we've seen, then we should stop.  So we're going to keep track of the minimum loss. And so let's just initially set that to infinity. 
It's a nice big number. Well, not quite a number,   but a number-ish like thing.  So then after every batch, first of  all, let's check that we're training. 
If we're not training, then  we don't want to do anything.  We don't use the learning  rate finder during validation. 
So here's a really handy thing. Just raise CancelEpochException,   and that stops it from doing that epoch entirely. So just to see how that works,  
you can see here one_epoch  does with the callback context 
manager epoch, and that will  say, oh, it's got canceled.  So it goes straight to the except, which is  going to go all the way to the end of that 
code and it's going to skip it.  So you can see that we're using exceptions  as control structures, which is actually a 
really powerful programming technique that  is really underutilized, in my opinion. 
Like a lot of things I do, it's  actually somewhat controversial. 
Some people think it's a bad idea, but I find  it actually makes my code more concise and 
more maintainable and more powerful. So I like it.  So let's see. So we've got our CancelEpochException. 
So then we're just going to keep  track of our learning rates.  The learning rates, we're going to learn a  lot more about optimizers shortly, so I won't 
worry too much about this.  But basically the learning rates are  stored by PyTorch inside the optimizer. 
And they're actually stored in  things called parameter groups.  So don't worry too much about the details,  but we can grab the learning rate from that 
dictionary. And we'll learn more about that shortly.  We've got to keep track of the loss,  append it to our list of losses. 
And if it's less than the minimum we've  seen, then record it as the minimum.  And if it's greater than three times  the minimum, then look at this. 
This is really cool. CancelFitException.  So this will stop everything  in a very nice, clean way. 
No need for lots of returns and  conditionals or stuff like that.  Just raise the CancelFitException. And then finally, we've got to actually update  
our learning rate to 1.3 times the previous one.  And so basically the way you do it in PyTorch  is you have to go through each parameter group 
and grab the learning rate in the  dictionary and multiply it by lr_mult. 
So yeah, you've already seen it run. And we can, at the end of running, you will  
find that there is now a, the callback will now contain an lrs and a losses. 
So for this callback, I can't just  add it directly to the callback list.  I need to instantiate it first. And the reason I need to instantiate it first  
is because I need to be able to grab its learning rates and its losses.  And in fact, you know, we could grab  that whole thing and move it in here. 
There's no reason callbacks only have  to have the callback things, right?  So we could do this. 
And now that's just going to become self. 
There we go. And so then we can train it   again and we could just call lrfind.plot. So callbacks can really be, you know,  
quite self-contained nice things, as you can see. So there's a more sophisticated callback and I  
think it's doing a lot of really nice stuff here.  You might've come across something in  PyTorch called learning rate schedulers. 
Learning Rate scheduler
And in fact, we could implement this whole  thing with a learning rate scheduler.  It won't actually save that much time. But I just want to show you when you use stuff  
in PyTorch like learning rate schedulers, you're actually using things that   are extremely simple. The learning rate scheduler  
basically does this one line of code for us. So I'm going to now create a new LRFinderCB. 
And this time I'm going to use the PyTorch's  ExponentialLR scheduler, which is here. 
So this is coming. It's interesting that actually the   documentation of this is kind of actually wrong. It claims that it decays the learning  
rate of each parameter group by gamma. So gamma is just some number you pass in.  I don't know why this has to be a Greek letter,  but it sounds more fancy than multiplying 
by an LR multiplier. It says every epoch.  But it's not actually done every epoch at all. What actually happens is, in PyTorch,  
the schedulers have a step  method and the decay happens each 
time you call step. And if you set gamma, which is actually lr_mult,  
to a number bigger than one, it's not a decay. It's an increase.  So the difference now, I guess I'll  copy and paste the previous version. 
Okay, so the previous version is on the top. So the main difference here is that before_fit,  
we're going to create something called a self.sched equal to the scheduler. 
And the scheduler, because it's going to be  adjusting the learning rates, it actually  needs access to the optimizer.  So we pass in the optimizer and  the learning rate multiplier. 
And so then in after_batch, rather than having  this line of code, we replace it with this 
line of code, self.sched.step(). So that's the only difference.  And you know, I mean, we're not gaining much,  as I said, by using the PyTorch ExponentialLR 
scheduler, but I mainly wanted to do it so  you can see that these things like PyTorch  schedulers are not doing anything magic. They're just doing that one line of code for us. 
And so I run it again using this new version. Oopsy-Daisey. 
Oh, I forgot to run this line of code. 
There we go. And I guess I should   also add the nice little plot method. Maybe we'll just move it to the bottom. 
There. 
lrfind.plot.  There we go. Put that one back to how it was. 
All right. Perfect timing.  So we added a few very important things in here. So make sure we export. 
And we'll be able to use them shortly.  All right. Let's have an 8-minute break. 
Let's just have a 10-minute break. So I'll see you back here at 8 past. 
All right. Welcome back.  One suggestion, which I really like,  is we could rename plot to after_fit,  
which I really  like, because that means we should be able  to then just call learn.fit and delete the 
next one. And let's see.  That didn't work. Why not? 
Oh, no. That doesn't work, does it?  Because the... you know what? I think the callback here  
could go into a finally block, actually. 
That would actually allow us to always  call the callback, even if we've canceled. 
I think that's reasonable. That may have its own confusions.  Anyway, we could try it for now, because  that would let us put this after_fit in. 
There we go. So that  
automatically runs that. So that's an interesting idea. 
I think I quite like it. Cool. 
Notebook 10
So let's now look at notebook 10. So I feel like this is the next big piece we need. 
So we've got a pretty good  system now for training models. 
What I think we're really missing, though, is  a way to identify how our models are training. 
And so to identify how our models are training,  we need to be able to look inside them and  see what's going on while they train. We don't currently have any way to do that. 
And therefore, it's very hard for  us to diagnose and fix problems. 
Most people have no way of  looking inside their models.  And so most people have no way to  properly diagnose and fix models. 
And that's why most people, when they have a  problem with training their model, randomly  try things until something  starts hopefully working. 
We're not going to do that. We're going to do it properly.  So we can import the stuff that  we just created in the learner. 
And the first thing I'm going to do,  introduce now is a set_seed function.  We've been using torch.manual_seed before. We know all about RNGs, random number generators. 
set_seed function
We've actually got three of them.  PyTorch’s, NumPy's and Python's. Let's see all of them. 
But also in PyTorch, you can use a flag to  ask it to use deterministic algorithms so 
things should be reproducible. As we discussed before, you shouldn't   always just make things reproducible. But for lessons, I think this is useful. 
So here's a function that lets  you set a reproducible seed.  All right, let's use the same data set  as before, a fashion MNIST dataset. 
Fashion-MNIST Baseline
We'll load it up in the same way. And let's create a model that  
looks very similar to our previous models. This one might be a bit bigger, mightn't it?  I didn't actually check. Okay. 
So let's use MultiClassAccuracy again. Same callbacks that we used before. 
We'll use the TrainCB version  for no particular reason.  And generally speaking, we want to train as  fast as possible, not just because we don't 
like wasting time, but actually more importantly  because the higher the learning rate you train  at, the more you're able to find, often,  a more generalizable set of weights. 
Training quickly also means that we can  look at each batch, each item in the  
data less often.  So we're going to have less  issues with overfitting.  And generally speaking, if we can train at  a high learning rate, then that means that 
we're learning to train in a stable way. And stable training is very good. 
So let's try setting up a high learning  rate of 0.6 and see what happens. 
So here's a function that's just going to  create our Learner with our callbacks and 
fit it and return the Learner  in case we want to use it.  And it's training up and  then it suddenly fell apart. 
So it's going well for a while and  then it stopped training nicely.  So one nice thing about this graph is that  we can immediately see when it stops training 
well, which is very useful. 
So what happened there? Why did it go badly?  I mean, we can guess that it might've been  because of our high learning rate, but what's 
really going on? So let's try to look inside it.  So one way to look inside it would be  we could create our own SequentialModel. 
1 - Look inside the Model
Just like the sequential model we've built before. Do you remember we created one using  
nn.ModuleList in a previous lesson? If you've forgotten, go back and check that out. 
And when we call that model, we go through  each layer and just call the layer. 
And what we could do is we could add something  in addition, which is at each layer, we could 
also get the mean of that layer and the standard  deviation of that layer and append them to 
a couple of different lists and activation  means and activation standard deviations. 
This is going to contain, after we call this  model, it's going to contain the means and 
standard deviations for each layer. And then we could define dunder iter,  
which makes this into an iterator, as being, let's say just, oh, just when you iterate through   this model, you can iterate through the layers. So we can then train this model in the usual way. 
And this is going to give us exactly the same  outcome as before, because I'm using the same  seed. So you can see it looks identical. 
But the difference is instead  of using nn.Sequential,   we've now used something that's actually saved the means and standard deviations of each layer. 
And so therefore we can plot them. 
Okay, so here we've plotted the activation means. And notice that we've done it for every batch. 
So that's why along the x-axis here we have  batch number and on the y-axis we have the 
activation means and then  we have it for each layer.  So rather than starting at one, because we're  starting at zero, so this is the first layer 
here's blue, second layer is  orange, third layer green,  
fourth layer red, and fifth layer white with that like mauve-y kind of color. 
And look what's happened. The activations have started pretty small, close  
to zero, and have increased at an exponentially increasing rate, and then have crashed, and then  
have increased again at an exponentially rate and crashed again.  And increased again, crashed again. And each time they've gone up, they've gone  
up even higher, and they've crashed, in this case even lower.  And what happens? Well, what's happening here  
when our activations are really close to zero? Well when your activations are really close to   zero, that means that the inputs to each layer are numbers very close to zero. 
As a result of which, of course,  the outputs are very close to zero.  Because we're doing just matrix multiplies. And so this is a disaster. 
When activations are very close  to zero, they're dead units. 
They're not able to do anything. And you can see for ages   here it's not training at all. And this is the activation means. 
The standard deviations  tell an even stronger story.  So generally speaking, you want the means  of the activations to be about zero, and the 
standard deviations to be about one. Mean of zero is fine as long   as they're spread around zero. But a standard deviation of close to zero is  
terrible, because that means all of the activations are about the same.  So here, after batch 30, all of the activations  are close to zero, and all of the standard 
deviations are close to zero, so all the numbers  are about the same, and they're about zero.  So nothing's going on. And you can see the same things  
happening with standard deviations. You start with not very   much variety in the weights. It exponentially increases how much  
variety there is, and then it crashes again. Exponentially increases, crashes again.  This is a classic shape of bad behavior. And with these two plots, you can really  
understand what's going on in your model. And if you train a model, and at the end of it,  
you kind of think, well, I wonder if this is any good.  If you haven't looked at this plot, you don't  know, because you haven't checked to see whether 
it's training nicely. Maybe it could be a lot better.  If you can get something, we'll see some nicer  training pictures later, but generally speaking, 
you want something where your mean is always  about zero, and your variance is always about 
one. Standard deviation is always about one.  And if you see that, then it's a pretty  good chance you're training properly. 
If you don't see that, you're most  certainly not training properly.  Okay, so what I'm going to do in the rest  of this part of the lesson is explain how 
to do this in a more elegant way. Because as I say, being able to look inside   your models is such a critically important thing to building and debugging models. 
We don't have to do it manually. We don't have to create our own sequential model.  We can actually use a PyTorch thing called hooks. So as it says here, a hook is called when a layer  
PyThorch hooks
that it's registered to is executed during the forward pass.  That's called a forward hook, or the backward  pass, and that's called a backward hook. 
And so the key thing about hooks is  we don't have to rewrite the model.  We can add them to any existing model. So we can just use standard nn.Sequential,  
passing in our layers, which were these ones here. 
And so we're still going to have something  to keep track of the activation means and  standard deviation. So just create an empty list, for now,  
for each layer in the model.  And let's create a little function that's  going to be called because a hook is going 
to call a function when during the forward  pass for a forward hook or the backward pass 
or backward hook. So it's got a function called append_stats.  It's going to be passed the hook number,  sorry, the layer number, the module,  
and the input and the output.  So we're going to be grabbing the outputs  mean and putting in activation means and the 
output standard deviation and putting  it in activation standard deviations.  So here's how you do it. We've got a model. 
You go through each layer of the model  and you call on it register_forward_hook.  That's part of PyTorch. And we don't need to write  
it ourselves because we already did, right? It's just doing the same thing as this basically. 
And what function is always going to be called? The function that's going to be called is  
the append_stats function passing in, remember partial is the equivalent of saying append_stats  
passing in i as the first element, the first argument. 
So if we now fit that model,  it trains in the usual way.  But after each layer, it's going to call this. 
And so you can see we get  exactly the same thing as before.  So one question we get here is what's the  difference between a hook and a callback? 
Nothing at all. Hooks and callbacks are the same thing.  It's just that PyTorch defines hooks and  they call them hooks instead of callbacks. 
They are less flexible than  the callbacks that we used  
in the Learner because you don't have access to all the available   states, you can't change things. But they're a particular kind of callback. 
It's just setting a piece of code that's  going to be run for us when something happens. 
And in this case, there's something that happens  is that either a layer in the forward pass  is called or a layer in the  backward pass is called. 
I guess you could describe the function that's  being called back as the callback and the 
thing that's doing the callback as the hook.  I'm not sure if that level of distinction  is important, but maybe you could do that. 
So anyway, this is a little bit fussy of creating  globals and appending to them and stuff like 
that. So let's try to simplify this a little bit.  So what I did here was I  created a class called Hook. 
So this class, when we create it, we're going  to pass in the module that we're hooking. 
So we call m.register_forward_hook  and we call the function.  We pass the function that we want to be given. And so here's the, we pass the function and we're  
also going to pass in the Hook class to the function. 
Let's also define a remove because this is  actually the thing that, this is actually 
the thing that removes the hook. We don't want it sitting around forever. 
This is called, dunder del is called  by Python when an object is freed.  So when that happens, we should  also make sure that we remove this. 
Okay. So append_stats now we're going to replace,   it's going to instead get past the Hook instead, because that's what we asked to be passed. 
And if there's no .stats attribute  in there yet, then let's create one. 
And then we're going to be passed the activations.  So put that on the CPU and append  the mean and the standard deviation. 
And now the nice thing is that the stats are  actually inside this object, which is convenient. 
So now we can do exactly the same thing as  before, but we don't have to set any of that  global stuff or whatever. We can just say, okay, our hooks is a Hook  
with that layer and that function for all those models layers. 
And so just calling it has called  register_forward_hook for us. 
So now when we fit that, it's  going to run with the hooks. 
There we go. It trains.  We need to do it too. 
So then it trains and we get exactly the same  shape as usual and we get back the same results  as usual. But as we can see,  
we're gradually making this  more convenient, which is nice. 
So we can make it nicer still because generally  speaking, we're going to be adding multiple 
hooks and this stuff of, you  know, this list comprehension,   whatever, it's a bit inconvenient. So let's create a Hooks class. 
Hooks class / context managers
So first of all, we'll see how  the Hooks class works in practice.  So in the Hooks class, the way we're going  to use it is we're going to call with Hooks, 
pass in the model, pass in the function to  use as our hook, and then we'll fit the model 
and that's it. It’s going to be literally just   one extra line of code to set up the whole thing. And then when we then… you can then go through  
each Hook and plot the mean and standard deviation of each layer. 
So that's how that's the hooks class  is going to make things much easier.  So the Hooks class, as you can see, we're  using a, making it a context manager. 
And we want to be able to loop through it. We want to be able to index into it. 
So it's quite a lot of behavior we want, believe  it or not, all that behavior is in this tiny 
little thing. And we're going to use the most   flexible general way of creating context managers. Now context managers are things that we can say  
“with”. The general way of creating a context manager is to create a class and define two  
special things dunder enter and dunder exit. dunder enter is a function that's going to  
be called when it hits the with statement. And if you add an as blah after it, then the  
contents of this variable will be whatever is returned from dunder enter. 
And as you can see, we just  returned the object itself.  So the Hooks object is  going to be stored in hooks. 
Now interestingly, the hooks  class inherits from list.  You can do this, you can actually  inherit from stuff like list in Python. 
So a Hooks, the Hooks object is a list. And therefore we need to call the  
super classes constructor. And we're going to pass in a,   that list comprehension we saw,  that list of hooks, where it's going 
to hook into each module in the list  of modules we asked to hook into. 
Now we're passing in a model here, but because  the model is an nn.Sequential, you can actually 
loop through an nn.Sequential and  it returns each of the layers.  So this is actually very, very  nice and concise and convenient. 
So that's the constructor. dunder enter just returns it.  dunder exit is what's called automatically  at the end of the whole block. 
So when this whole thing's finished,  it's going to remove the hooks.  And removing the hooks is just going  to go through each Hook and remove it. 
The reason we can do “for h in self”  is because, remember, this is a list. 
And then finally we've got  a dunder del like before.  And I also added a dunder delitem. This is the thing that lets you delete a  
single Hook from the list which will remove that one Hook and call the list's __delitem__. 
So there's our whole thing. This one's optional.  This is the one that lets us remove a  single Hook rather than all of them. 
So let's just understand some  of what's going on there.  So here's a dummy context manager. As you can see here, it's got a dunder enter,  
Dummy context manager, Dummy list
which is going to return itself and it's going to print something.  So you can see here I call with DummyCtxMgr. And so therefore it prints “let's go!” first. 
The second thing it's going to do is call  this code inside the context manager. 
So we've got “as dcm”, so that's itself. And so it's going to call hello,   which prints “hello.” So here it is. 
And then finally it's going to automatically  call exit, dunder exit, which is “all done!”. 
So here's “all done!”. So again, if you haven't used context managers   before, you want to be creating little samples like this yourself and getting them to work. 
So this is your key homework for this week. Is anything in the lesson where we're using a  
part of Python you're not a hundred percent familiar with is for you to, from scratch,   to create some simple like kind of dummy version that fully explores what it's doing. 
If you're familiar with all the  Python pieces, then it's to create   your own, you know, that is to explore, do the same thing with  
the PyTorch pieces, like with, with hooks and so forth. 
And so I just wanted to show you also  what it's like to inherit from list.  So here I'm here inheriting from a list and  I could redefine how dunder delitem works. 
So now I can create a DummyList and it looks  exactly the same as usual, but now if I delete 
an item from the list, it's going  to call my overridden version  
and then it will call the original version.  And so the list is now got removed that  item and did this at the same time. 
So you can see, you can actually, yeah, modify  how Python works or create your own things 
that get all the behavior or the convenience  of Python classes like this one and add stuff 
to them. So that's what's happening there.  Okay. So that's our hooks class. 
So the next bit was developed, largely developed,  the last time I think it was that we did a 
Part 2 course in San Francisco with Stefano. So many thanks to him for helping get  
this next bit looking great. We're going to create my favorite  
Colorful Dimension: histogram
single image explanations of  what's going on inside a model. 
We call them the colorful  dimension, which they're histograms. 
We're going to take our same append_stats. These are all the same as before.  We're going to add an extra line of code,  which is to get a histogram of the absolute 
values of the activations. So a histogram, to remind you, is something  
that takes a collection of numbers and tells you how frequent each group of numbers are. 
And we're going to create  50 bins for our histogram. 
So we will use our hooks that we just  created, and we're going to use this  
new version of append_stats.  So it's going to train as before, but now  we're going to, in addition, have this extra 
thing in stats, which is  going to contain a histogram. 
And so with that, we're now going  to create this amazing plot. 
Now what this plot is showing  is for the first, second, third,   and fourth layers, what does the training look like? 
And you can immediately see the basic idea  is that we're seeing the same pattern. 
But what is this pattern showing? What exactly is going on in these pictures? 
So I think it might be best if we  
try and draw a picture of this. So let's take a normal histogram. 
So let's take a normal  histogram where we basically  
have grouped all the data into bins,  and then we have counts of  how much is in each bin. 
So for example, this will be like the value  of the activations, and it might be, say, 
from 0 to 10, and then from  10 to 20, and from 20 to 30.  And these are generally equally spaced bins. Okay. 
And then here is the count. So that's the number of items  
with that range of values. So this is called a histogram. 
Okay. So what Stefano and I did was we actually  
turned that histogram, that whole histogram,  into a single column of pixels. So if I take one column of pixels,  
that's actually one histogram. And the way we do it is we take these numbers. 
So let's say it's like 14, that  one's like 2, 7, 9, 11, 3, 2, 4, 2. 
And so then what we do is we  turn it into a single column. 
And so in this case we've got 1, 2, 3,   4, 5, 6, 7, 8, 9 groups, right? So we would create our 9 groups. 
Sorry, they were meant to be evenly  spaced, but they were not a very good job.  Got our 9 groups. And so we take the first group, it's 14. 
And what we do is we color it with a gradient  and a color according to how big that number 
is. So 14 is a real big number.  So depending on what gradient we  use, maybe red's really, really big. 
And the next one's really small,  which might be like green.  And then the next one's quite big  in the middle, which is like blue. 
Next one's getting quite, quite bigger still. So maybe it's just a little bit, sorry,   should go back to red. Go back to more red. 
Next one's bigger stills, it's even more red   and so forth. So basically we're taking the histogram  
and taking it into a color coded single column plot, if that makes sense. 
And so what that means is that at the  very, so let's take layer number two here. 
Layer number two, we can  take the very first column.  And so in the color scheme that  actually Matplotlib's picked here,  
yellow is the most common and then light green is less common.  And then light blue is less  common and then dark blue is 0. 
So you can see the vast majority is 0 and  there's a few with slightly bigger numbers,  which is exactly the same that  we saw for index one layer. 
Here it is, right? The average  
is pretty close to 0. The standard deviation is pretty small.  This is giving us more information, however. So as we train at this point here, there is  
quite a few activations that are a lot larger, as you can see.  And still the vast majority  of them are very small. 
There's a few big ones, they've still  got a bright yellow bar at the bottom.  The other thing to notice here is what's happened  is we've taken those stats, those histograms, 
we've stacked them all up into a single  tensor, and then we've taken their log.  Now log1p is just log of the number plus one. That's because we've got zeros here. 
And so just taking the log is going to kind of   let us see the full range more clearly. So that's what the log's for. 
So basically what we'd really ideally like  to see here is that this whole thing should 
be a kind of more like a rectangle. The maximum should be not changing very much. 
There shouldn't be a thick yellow bar at the  bottom, but instead it should be a nice even  gradient matching a normal distribution. Each single column of pixels wants to be kind  
of like a normal distribution, so gradually decreasing the number of activations. 
That's what we're aiming for. There's another really important and actually  
easier to read version of this, which is what if we just took those first two bottom pixels,  
so the least common 5%, and counted up how many were in, sorry, the least common 5%. 
The least common, not least  common either, let's try again. 
In the bottom two pixels, we've got the smallest  two equally sized groups of activations. 
We don't want there to be too many of them  because those are basically dead or nearly 
dead activations. They're much,   much, much smaller than the big ones. And so taking the ratio between those bottom  
two groups and the total basically tells us what percentage have zero or near  
zero or extremely small magnitudes. And remember that these are with absolute values. 
So if we plot those, you can see how bad this is. And in particular, for example, at the final  
layer, nearly from the very start, really, nearly all of the activations are  
just about entirely disabled. So this is bad news. 
And if you've got a model where most of  your model is close to 0, then most of your 
model is doing no work. And so it's really not working. 
So it may look like at the very  end, things were improving.  But as you can see from  this chart, that's not true. 
The vast majority are still inactive. Generally speaking, I found that if early  
in training you see this rising crash, rising crash at all, you should stop and restart training  
because your model will probably never recover. 
Too many of the activations  have gone off the rails.  So we want it to look kind of like this the  whole time, but with less of this very thick 
yellow bar, which is showing us most are inactive. 
Okay. So that's our activations. 
So we've got really now all of the kind of key  pieces I think we need to be able to flexibly 
change how we train models and to understand  what's going on inside our models. 
And so from this point, we've kind of like  drilled down as deep as we need to go and 
we can now start to come back up again and  put together the pieces, building up what 
are all of the things that are going to  help us train models reliably and quickly. 
And then hopefully we're going to be able to  successfully create from scratch some really  high quality generative models  and other models along the way. 
Okay. I think   that's everything for this class. But next class, we're going to start   looking at things like initialization. It's a really important topic. 
If you want to do some revision before then  just make sure that you're very comfortable  with things like standard deviations and stuff  like that because we're using that quite a 
lot for next time. And yeah, thanks for joining me. 
Look forward to the next lesson. See you again.
