# The-Fitness-Coach
ECE445 Senior Design Project UIUC SP25

## PROBLEM

With the national fitness campaign, people's fitness habits are being established, and the demand for personalized fitness plans is increasing. Many individuals now exercise at home due to its convenience. However, unfamiliarity with professional knowledge can lead to injuries. This is the issue the Smart Fitness Coach aims to solve. Our goal is to provide an accessible and customized fitness experience, enhancing workout safety and efficiency.

## SOLUTION OVERVIEW

Our app, The Smart Fitness Coach helps home exercisers by offering a system that recognizes exercise forms, provides real-time feedback on posture, and suggests improvements for better safety and effectiveness based on the captured movements.

## SOLUTION COMPONENTS

### FRONTEND (MOBILE APPLICATION)

User Interface: A user-friendly interface that provides visual feedback on exercise form, indicating correct and incorrect posture.
Real-Time Feedback: Visual cues and alerts to guide users, such as highlighting misaligned body parts and offering feedback on exercise correctness.
Exercise Suggestions: Personalized workout recommendations based on user performance.

### BACKEND (PROCESSING UNIT)

Data Processing: Real-time processing of the video feed, ensuring minimal delay in feedback to the mobile app during workouts.
Pose Estimation: Techniques like OpenPose or MediaPipe are used to estimate the user’s body pose and evaluate alignment during exercises.
Action Recognition: Machine learning models, for example, Yolo, identify exercises such as squats or push-ups by analyzing movement patterns.

### CLOUD DATABASE (OPTIONAL)

Stores user profiles, exercise logs, and performance metrics. The cloud also hosts models, enabling periodic updates to improve accuracy.

## CRITERION OF SUCCESS

The project will be successful if the system can accurately identify user movements, provide reliable feedback, and suggest improvements to reduce injuries and enhance fitness. User satisfaction and fitness improvement are key indicators of success.
