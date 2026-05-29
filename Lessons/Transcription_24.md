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
