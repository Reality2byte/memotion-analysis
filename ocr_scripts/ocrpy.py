##def implicit():
##    from google.cloud import storage
##
##    # If you don't specify credentials when constructing the client, the
##    # client library will look for credentials in the environment.
##    #project = 'my_project_name'
##    storage_client = storage.Client.from_service_account_json('C:\\Users\\harsh\\Downloads\\memotion_analysis\\ocr\\ocrenv\\gcred.json')
##
##    # Make an authenticated API request
##    buckets = list(storage_client.list_buckets())
##    print('buckets',buckets)

##implicit()

def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
##    client = vision.ImageAnnotatorClient()
##    storage_client = storage.Client.from_service_account_json('C:\\Users\\harsh\\Downloads\\memotion_analysis\\ocr\\ocrenv\\gcred.json')

    client = vision.ImageAnnotatorClient.from_service_account_file(
   'C:\\Users\\harsh\\Downloads\\memotion_analysis\\ocr\\ocrenv\\gcred.json')

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

detect_text('C:\\Users\\harsh\\Downloads\\memotion_analysis\\ocr\\ocrenv\\avengers_1pd1hg.jpg')
