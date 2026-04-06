import models
import make_video
import user_interaction as user

print('Hi, please choose the model you want to investigate\nEnter corresponding number:')
print('linear oscillator - 1, math pendulum - 2')

model_num: int = user.get_number()

if model_num == 1:
    properties = user.get_model_properties()
    print('Now we are going to investigate linear oscillator, please wait ..')
    make_video.create_video(properties, models.oscillation, make_video.create_frame_oscillation)
    user.run_video(properties, models.oscillation)


if model_num == 2:
    properties = user.get_model_properties()
    print('Now we are going to investigate math pendulum, please wait ..')
    make_video.create_video(properties, models.math_pendulum, make_video.create_frame_math_pendulum)
    user.run_video(properties, models.math_pendulum)

if not (model_num in [1, 2]):
    print('Opps, you\'ve failed to choose the model, rerun programm and look at model\'s numbers carefully')

