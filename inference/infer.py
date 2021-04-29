
def run_on_image(image):
    start_time = time.time()
    predictions, visualized_output = demo.run_on_image(image)
    time_elapsed = time.time() - start_time