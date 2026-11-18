# Limitations and ethics

SimCityNet measures what a street *looks like* through a handful of pretrained models. Read
the numbers with these limits in mind.

## The instrument has its own geography

The segmentation and detection models were trained on datasets collected largely in Europe and
North America; Cityscapes, for example, was recorded in German and neighbouring European
cities. They may label a souq canopy, a palm-lined median or a desert-coloured façade
less accurately than a European street. When comparing regions, part of any difference can come
from the models rather than the scenes. Inspect scene cards for a sample of each region before
drawing conclusions, and report this caveat.

## Measurement limits

- **Counts saturate and miss small objects.** Dense crowds, distant pedestrians and partly
  hidden vehicles are undercounted; reflections and posters can be counted as people.
- **Depth is relative.** DPT predicts relative inverse depth per image, so the vegetation
  weighting ranks pixels within one image; it is not metric distance.
- **Colour depends on light.** Time of day, weather, white balance and image compression move
  the palette as much as the city does. Compare images taken in similar conditions, or rely on
  the other blocks.
- **Synthetic frames can confuse models.** Generated videos contain artefacts (melting objects,
  impossible geometry) that segmenters and detectors handle unpredictably.
- **Panoramas vs photos.** Field of view changes every share. Do not mix panoramas and ordinary
  photos in one comparison without care.
- **Batch normalisation depends on the set.** With `normalize="batch"`, adding an image can
  change the similarity of other pairs. Use `normalize="fixed"` for results that must stay
  stable across runs.

## Responsible use

- **Not for surveillance.** The package counts people; it does not identify, track or store
  them, and must not be extended to do so on images of identifiable individuals without consent
  and ethical approval.
- **Image rights.** Respect the terms of your image sources. Google Street View imagery, used in
  the paper, may not be redistributed; the tutorials therefore use openly licensed photos from
  Wikimedia Commons with credits.
- **Appearance is not culture.** The embedding describes greenery, colour, activity and
  surfaces. It does not measure culture, wealth or safety, and differences should not be
  presented as such.
- **Model licences.** Some models carry non-commercial or copyleft terms; see
  [Model licences](model_licenses.md).
