* Remove hardcoded source dir on Google Drive.
* Add:

```
# New Similarity Threshold Widget (using L2 Distance)
similarity_threshold_widget = widgets.FloatSlider(
    value=0.4, # Default value set to 0.4
    min=0.0,
    max=2.0, # L2 distance can be > 1.0, 2.0 is a reasonable upper bound for normalized embeddings
    step=0.05,
    description='L2 Distance Threshold:', # Renamed description to reflect L2 distance metric
    style={'description_width': 'initial'}
)
```
See C:\Users\saint\Documents\ai-working\gemini-rag\multi-qa-distilbert-cos-v1-wo-ui-1-gd-8-1.py for location of parameter. 
