# BDD-Net
The timely and accurate recognition of damage to buildings after destructive disasters is
one of the most important post-event responses. Due to the complex and dangerous situations in
affected areas, field surveys of post-disaster conditions are not always feasible. The use of satellite
imagery for disaster assessment can overcome this problem. However, the textural and contextual
features of post-event satellite images vary with disaster types, which makes it difficult to use
models that have been developed for a specific disaster type to detect damaged buildings following
other types of disasters. Therefore, it is hard to use a single model to effectively and automatically
recognize post-disaster building damage for a broad range of disaster types. Therefore, in this
paper, we introduce a building damage detection network (BDD-Net) composed of a novel
end-to-end remote sensing pixel-classification deep convolutional neural network. BDD-Net was
developed to automatically classify every pixel of a post-disaster image into one of non-damaged
building, damaged building, or background classes. Pre- and post-disaster images were provided
as input for the network to increase semantic information, and a hybrid loss function that combines
dice loss and focal loss was used to optimize the network. Publicly available data were utilized to
train and test the model, which makes the presented method readily repeatable and comparable.
The protocol was tested on images for five disaster types, namely flood, earthquake, volcanic
eruption, hurricane, and wildfire. The results show that the proposed method is consistently
effective for recognizing buildings damaged by different disasters and in different areas.


See the details in the paper: https://doi.org/10.3390/rs12101670
