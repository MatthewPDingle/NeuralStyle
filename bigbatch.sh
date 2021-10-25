# Init
MODEL_FILE=models/VGG16_SOD_finetune.pth

TV_WEIGHT=0
SAVE_ITER=0
ORIGINAL_COLORS=1

PYTHON=python3 # Change to Python if using Python 2
SCRIPT=neural_style.py
GPU=0

NEURAL_STYLE=$PYTHON
NEURAL_STYLE+=" "
NEURAL_STYLE+=$SCRIPT

# Refactor Params - The multiple that each param gets changed by each iteration.
CONTENT_WEIGHT_REFACTOR=2.5
STYLE_WEIGHT_REFACTOR=4
STYLE_SCALE_REFACTOR=.8
IMAGE_SIZE_REFACTOR=1.6666 # Yields a 392 -> 3021 pixel image after the 5th iteration is complete.

# Loop Process
for IMAGE in "images/batch"/*
do
	for STYLE in "styles/batch"/*
	do
		echo $IMAGE, $STYLE
		# Initial Params for first iteration.
		CONTENT_WEIGHT=5
		STYLE_WEIGHT=2
		STYLE_SCALE=.5	# 1 Default
		IMAGE_SIZE=392
		NUM_ITER=30
		BACKEND=$"cudnn -cudnn_autotune"
		OPTIMIZER=lbfgs

		for i in {1..5};
		do
			OUT_NAME=out$i.png
			INIT=random
			INIT_IMAGE=""
			if [ $i -gt 1 ]; then 
				INIT=image
				INIT_IMAGE=out$((i-1)).png
			else
				INIT_IMAGE=$IMAGE
			fi
			if [ $i -ge 5 ]; then
				IMAGE_NAME=${IMAGE##*/}
				IMAGE_NAME=${IMAGE_NAME%%.*}
				STYLE_NAME=${STYLE##*/}
				STYLE_NAME=${STYLE_NAME%%.*}
				OUT_NAME=$IMAGE_NAME$"_"$STYLE_NAME$".png"
				NUM_ITER=150
				#BACKEND=cudnn
				OPTIMIZER=adam
			elif [ $i -ge 4 ]; then
				NUM_ITER=400
				#BACKEND=cudnn
			fi

			# Call the python script
			$NEURAL_STYLE \
		  	-content_image $IMAGE \
			-style_image $STYLE \
			-model_file $MODEL_FILE \
			-init $INIT \
			-init_image $INIT_IMAGE \
			-print_iter 0 \
			-save_iter $SAVE_ITER \
			-content_weight $CONTENT_WEIGHT \
			-style_weight $STYLE_WEIGHT \
			-style_scale $STYLE_SCALE \
			-image_size $IMAGE_SIZE \
			-num_iterations $NUM_ITER \
			-output_image $OUT_NAME \
			-tv_weight $TV_WEIGHT \
			-original_colors $ORIGINAL_COLORS \
			-backend $BACKEND \
			-optimizer $OPTIMIZER
			
			# Update some params for the next iteration
			CONTENT_WEIGHT=`echo $CONTENT_WEIGHT*$CONTENT_WEIGHT_REFACTOR|bc -l`
			STYLE_WEIGHT=`echo $STYLE_WEIGHT*$STYLE_WEIGHT_REFACTOR|bc -l`
			STYLE_SCALE=`echo $STYLE_SCALE*$STYLE_SCALE_REFACTOR|bc -l`
			IMAGE_SIZE=`echo $IMAGE_SIZE*$IMAGE_SIZE_REFACTOR|bc -l`
			IMAGE_SIZE=`echo ${IMAGE_SIZE%.*}`
		done	
	done
done
