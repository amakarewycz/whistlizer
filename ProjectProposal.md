My project is about building a classifier to detect whistles from the refereeing officials in volleyball matches.

I have access to over 100 videos of my daughters High School matches.  And I will be including their NorCal D1 Championship 
game in the training set.  To ensrine that win into an ML model.

Why would one do this?

Navigating videos is tedious.  With whistle detection one could derive useful information that could eliminate the bordom 
of manually finding play starts.  But there are other benefits.

Automate tallying of match score. Machine can keep track of score by counting whistles.

Align score tally with hyperlinks to start of playing video for each point

Highlight interesting plays in the match.  Like extended rallies where a team comes back from behind after losing their
mental edge.

Business Case:
I think there is potential of creating a business out of such a machine.  Coaches can use it to make their "chalk talk"
sessions slick.  They can wear their iWatch and expertly click the playlist without having to fumble with video 
fast forward / rewind controls when reviewing plays.  But thats not all.  Parents and Athetes could also use this machine.

This capability could also help Althetes in their quest to be recruited by major College and Universities. 
An prospective althete could make it easier for prospective colleges to view their skills by sending playlists & metadata
generated from this machine.

How would it work;

The input data will be a video of the volleyball match. 

The output will be a time series of predictions: 
1. if a serve is in play, for a given time interval 
2. which team is serving.

Use the output to enrich the video with cue points, running game score, and exciting rally plays as described above.

A web app will be used to present the results. 

The webapp will have input fields to specify various parameters that impact the predictions: probability_threshold, frame_threshold
It will have several widgets: Video Upload, Video Player, Buttons for each Cue Point, File Upload for loading Ground Truth, Confusion Matrix

Immediately after upload, the video will be featurized and sent to the model for inference by the webapp.  

Underneath the video will be a table with following columns. 
1. A column of hyperlinks for that cue the video to start of a serve for the next point.
2. A column with match score. 
3. A column to indictate exciting rally play. The table could also be replaced with a graphic.
The graphic could be a timeseries of excitement level; plus vertical marks to indicate when service starts for each
point.

If there is a ground truth file, the web app can compare the outcomes vs. ground truth.  It can display a confusion matrix.
It can also update the cue points and mark each outcome: TP, FP, TN, FN.

