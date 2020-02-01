#!/usr/bin/env bash


TIMEOUT=300     # timeout in seconds before a new simulation is started
NUM_SIMS=2      # number of simulation
CURRENT_SIM=0   # counter for simulation
TIME=0          # time keeper of simulation


while [ $CURRENT_SIM -le $NUM_SIMS ]
do
  sleep 10
  echo "current simulation: ${CURRENT_SIM} of ${NUM_SIMS}"
  ((CURRENT_SIM++))

  FCU_URL="udp://:14540@127.0.0.1:14557"
  TITLE1="PX4"
  TITLE2="MAVROS"
  TITLE3="DOMAIN_RANDOMISATION"
  TITLE4="execute_trajectories"
  FCU_URL="udp://:14540@127.0.0.1:14557"
  MAVROS="roslaunch mavros px4.launch fcu_url:=${FCU_URL}"
  CMD_PX4="make px4_sitl HEADLESS=1 gazebo_honeybee"
  #CMD_PX4="make px4_sitl gazebo_honeybee"

  CMD1="cd /home/henry/esl-sun/PX4/ && ${CMD_PX4}"
  CMD2=${MAVROS}
  CMD3="rosrun domain_randomisation domain_randomisation"
  CMD4="rosrun px4_nav_cmd execute_trajectories.py -n=1"


  SEARCH_CMD1="/home/henry/esl-sun/PX4/build/px4_sitl_default/bin/px4 /home/henry/esl-sun/PX4/ROMFS/px4fmu_common -s etc/init.d-posix/rcS -t /home/henry/esl-sun/PX4/test_data"
  SEARCH_CMD2="/usr/bin/python /opt/ros/melodic/bin/roslaunch mavros px4.launch fcu_url:=udp://:14540@127.0.0.1:14557"
  SEARCH_CMD3="home/henry/catkin_ws/devel/lib/domain_randomisation/domain_randomisation"
  SEARCH_CMD4="python /home/henry/catkin_ws/src/px4_nav_cmd/scripts/execute_trajectories.py -n=1"

  xterm -e "bash -c '${CMD1}'" &
  sleep 5
  xterm -e "bash -c '${CMD2}'" &
  sleep 5
  xterm -e "bash -c '${CMD3}'" &
  sleep 5
  xterm -e "bash -c '${CMD4}'" &

  # Retrieves all the infomations regarding the programs processes
  sleep 10;
  PX4_INFO=`ps aux | grep "${SEARCH_CMD1}"`
  MAVROS_INFO=`ps aux | grep "${SEARCH_CMD2}"`
  DR_INFO=`ps aux | grep "${SEARCH_CMD3}"`
  PY_INFO=`ps aux | grep "${SEARCH_CMD4}"`


  # echo "PX4_INFO: " $PX4_INFO
  # echo " "
  # echo "MAVROS_INFO: " $MAVROS_INFO
  # echo " "
  # echo "DR_INFO: " $DR_INFO
  # echo " "
  # echo "PY_INFO: " $PY_INFO


  PX4_array=($PX4_INFO)

  PX4_PID=${PX4_array[1]}         #PID of the program
  PX4_START_TIME=${PX4_array[8]}  #Time at which the program was started
  PX4_TOTAL_TIME=${PX4_array[9]}  #Total time which the program has been running

  # echo "PX4 PID: " $PX4_PID # Process Number
  # echo "PX4 START_TIME: " $PX4_START_TIME # Time of how long the program has been active
  # echo "PX4 TOTAL_TIME: " $PX4_TOTAL_TIME #Process name according top, should be the same as $PROGRAM_NAME



  MAVROS_array=($MAVROS_INFO)
  MAVROS_PID=${MAVROS_array[1]}         #PID of the program
  MAVROS_START_TIME=${MAVROS_array[8]}  #Time at which the program was started
  MAVROS_TOTAL_TIME=${MAVROS_array[9]}  #Total time which the program has been running

  # echo "MAVROS PID: " $MAVROS_PID # Process Number
  # echo "MAVROS START_TIME: " $MAVROS_START_TIME # Time of how long the program has been active
  # echo "MAVROS TOTAL_TIME: " $MAVROS_TOTAL_TIME #Process name according top, should be the same as $PROGRAM_NAME



  DR_array=($DR_INFO)
  DR_PID=${DR_array[1]}         #PID of the program
  DR_START_TIME=${DR_array[8]}  #Time at which the program was started
  DR_TOTAL_TIME=${DR_array[9]}  #Total time which the program has been running

  # echo "DR PID: " $DR_PID # Process Number
  # echo "DR START_TIME: " $DR_START_TIME # Time of how long the program has been active
  # echo "DR TOTAL_TIME: " $DR_TOTAL_TIME #Process name according top, should be the same as $PROGRAM_NAME



  PY_array=($PY_INFO)
  PY_PID=${PY_array[1]}         #PID of the program
  PY_START_TIME=${PY_array[8]}  #Time at which the program was started
  PY_TOTAL_TIME=${PY_array[9]}  #Total time which the program has been running

  # echo "PY PID: " $PY_PID # Process Number

  TIMEKEEPER=`date +%s`
  # echo $TIMEKEEPER
  TIMEKEEPER=$((TIMEKEEPER+160))
  # echo $TIMEKEEPER
  export TIMEKEEPER
  export DR_PID
  export PY_PID   #Gives access to the subshell when we are in the while loop
  export MAVROS_PID
  export PX4_PID
  export PY_TOTAL_TIME   #Gives access to the subshell when we are in the while loop
  export SEARCH_CMD4
  sleep 2

  { while sleep 5; do #check every (x) seconds if the program has crash

      NOW_TIME=`date +%s`
      # echo "${NOW_TIME} ${TIMEKEEPER}"
      PY_INFO=`ps aux | grep "${SEARCH_CMD4}"`  #Find the information about the processes of the program
      array=($PY_INFO)
      NEW_PID=${array[1]}

      sleep 5                     #This can be removed if the sleep value above is long enough


      if [ "$PY_PID" != "$NEW_PID" ]
      then #if PID differ the program has terminate and we close all
        echo "simulation has ended"
        kill -SIGINT $MAVROS_PID
        kill -SIGINT $PX4_PID
        kill -SIGINT $DR_PID
        kill -SIGKILL $PY_PID
        break
      elif [ $TIMEKEEPER -lt $NOW_TIME ]
      then
        echo "simulation is broken, restarting"
        kill -SIGINT $MAVROS_PID
        kill -SIGINT $PX4_PID
        kill -SIGINT $DR_PID
        kill -SIGKILL $PY_PID
        break
      fi
    done }
done
