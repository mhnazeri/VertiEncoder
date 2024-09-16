#! /usr/bin/env bash

export PROJECT_NAME=vertiencoder  # add your project folder to python path
export PYTHONPATH=$PYTHONPATH:$PROJECT_NAME
export COMET_LOGGING_CONSOLE=info
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # adds 24M overhead to memory usage
# for torch.compile debugging
#export TORCH_LOGS="+dynamo"
#export TORCHDYNAMO_VERBOSE=1

Help()
{
   # Display Help
   echo 
   echo "Facilitates running different stages of training and evaluation."
   echo 
   echo "options:"
   echo "train                      Starts training."
   echo "eval                       Starts evaluation."
   echo "run file_name              Runs file_name.py file."
   echo
}

run () {
  case $1 in
    swae)
      python $PROJECT_NAME/train_swae.py --conf $PROJECT_NAME/conf/swae
      ;;
    train)
      python $PROJECT_NAME/train_pretext.py --conf $PROJECT_NAME/conf/transformer
      ;;
    train_dt)
      python $PROJECT_NAME/train_dt.py --conf $PROJECT_NAME/conf/dt
      ;;
    vanilla)
      python $PROJECT_NAME/train_vanilla_dt.py --conf $PROJECT_NAME/conf/vanilla
      ;;
    -h) # display Help
      Help
      exit
      ;;
    *)
      echo "Unknown '$1' argument. Please run with '-h' argument to see more details."
      # Help
      exit
      ;;
  esac
}

run $1 $2

echo "Done."
