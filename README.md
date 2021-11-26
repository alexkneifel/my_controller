# my_controller

ENPH 353 Project involving ROS, Gazebo, Python, OpenCV, PID controllers, and DNN.

 1. Players
    1. A team consists of 2 students
    2. Each team controls one robot
2. Goal
    1. Each team participates in a 4 minute (sim-time) round where they return license plates and locations
    2. For each correct license plate and location returned the team is awarded points
    3. Points are deducted for Traffic Rules violations
    4. The team with the highest score wins
3. Robot control
    1. The competition is taking place in simulation (Gazebo)
    2. A team is provided with a live image feed from the in-simulation camera mounted on top of the robot
    3. A team can send control commands to move the robot backwards/forwards, turn it CW/CCW or stop it
    4. A team can submit a message with (license plate id, location id) to ROS Master
        Correct plate id: +6pts (outside ring), +8pts (inside ring)
4. Traffic Rules
    1. Maintain at least 2 wheels inside the white lines marking the road (-2 pts)
    2. Do not collide with other robots (-5pts)
        NOTE: If the NPC truck rear-ends your robot no penalties are given
    3. Do not run over pedestrians (-10pts)
    4. Complete a full outside ring lap (+5pts)
5. Start/stop competition
    1. To start the timer for the competition send a message on the /license_plate topic with license plate ID 0 i.e. str('TeamRed,multi21,0,XR58')
    2. To stop the timer for the competition send a message to the /license_plate topic with license plate ID -1 (minus one) i.e str(‘TeamRed,multi12,-1,XR58’)
6. Other rules
    1. You must design and train your own neural networks. You cannot use pre-trained neural networks (i.e. Tesseract)
