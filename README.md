# CALLMSAE
Cascading Large Language Models for Salient Event Graph Generation (preprint)


## Step One

Initialize the directories

```
sh init_dir.sh
```

## Step Two

Save the NYT dataset as ```NYT_annotated``` in the directory. Run the following command

```
python get_nyt_data
```

## Step Three

```
python get_salient_events.py
```

## Step Four

```
python event_relation_prompting.py
```

Data preprocess code and training data are coming soon
