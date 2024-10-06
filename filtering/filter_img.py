import sglang as sgl
from sglang import RuntimeEndpoint
from pathlib import Path
from tqdm import tqdm
import pandas as pd

@sgl.function
def guard_gen(s, image_path, prompt):
  s += sgl.user(sgl.image(image_path) + prompt)
  hyperparameters = {
    'temperature': 0.2,
    'top_p': 0.95,
    'top_k': 50,
    'max_tokens': 500,
  }
  s += sgl.assistant(sgl.gen("json_output", **hyperparameters))

base_dir = "./images"
start_id = 100_000
im_path = base_dir + '00304/003049749.jpg'

prompt = """
Provide a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:
O1: Hate, Humiliation, Harassment: 
Should not:
- Contain content that expresses hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Contain content that encourages discrimination or harassment against people based on those characteristics.
Can:
- Contain educational content that provides information on discrimination, humiliation, and harassment.
- Discuss historical events involving hate and harassment.
O2: Violence, Harm, or Cruelty: 
Should not:
- Contain content of people planning or engaging in violence.
- Contain content that encourages people to engage in violence or cruelty, including harm to others.
Can:
- Contain educational content that provides information on violence, harm, and cruelty.
- Discuss historical events involving violence and cruelty.
O3: Sexual Content: 
Should not:
- Contain sexually explicit (i.e., erotic) content.
Can:
- Contain content that is related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
- Contain content of people in a loving and affectionate manner if it is not sexually explicit content.
O4: Nudity Content: 
Should not:
- Contain visible genitalia (e.g., exposed breast, nipple, vagina, vulva, or penis) or explicit nudity content. 
Can:
- Contain artistic nudity if the content does not include visible genitalia.
- Contain educational content and general information on the human body.
- Provide guidance on topics related to sexual health.
O5: Criminal Planning: 
Should not:
- Contain content of people planning or engaging in criminal activities like theft, kidnapping, or financial crimes.
- Contain content that encourages people to engage in criminal activities or condone criminal behavior.
Can:
- Contain content that helps people understand how crimes are committed, provided that the content is not so detailed that it could be used to provide a roadmap for future crimes.
- Contain content that educates on crime prevention and understanding criminal behavior without providing guidance for criminal activities.
O6: Weapons or Substance Abuse: 
Should not:
- Contain content of illegal weapons with the intent to promote violence or harm.
- Contain content of trafficking, consumption, or use of illegal drugs.
- Promote the use of tobacco, alcohol, and cannabis.
- Contain content about the creation or production of illegal substances.
Can:
- Contain weapons used in a legal and safe context without any indication of violence or intent to harm.
- Discussions on firearm ownership and educational content on regulated substances without promoting illegal activities.
- Contain content that provides factual information about legal drugs, including their effects and chemical composition, without promoting their use.
- Contain content that offers educational insights into the manufacturing processes of substances, ensuring that such information is general and does not include detailed instructions that could facilitate their production.
O7: Self-Harm: 
Should not:
- Contain content that could encourage people to harm themselves.
- Contain content that romanticizes or condones self-harm or suicide.
Can:
- Contain educational content that informs about the prevalence, causes, and effects of self-harm and suicide, such as statistical data and discussions on mental health challenges.
- Contain resources and information that guide individuals toward professional help and support services for mental health, self-harm, and suicide prevention.
O8: Animal Cruelty: 
Should not:
- Contain content that could encourage people to harm animals.
- Contain content that romanticizes or condones harming animals.
- Contain explicit animal cruelty.
Can:
- Contain content that provides general information regarding animal cruelty.
O9: Disasters or Emergencies: 
Should not:
- Contain content about individuals who are deceased or have suffered harm as a result of disasters or emergencies.
- Contain content about suffering, cruelty, or exploitation of individuals during such events.
Can:
- Contain respectful news coverage and reports on emergencies, prioritizing informational value over sensationalism.
- Contain educational content that provides factual information on historical and current disasters or emergencies.
Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Explain the reason why the content is considered safe or unsafe, referring to the description of the assigned policy category.
To provide your assessment use the following json template:
{
  "rating": str<"Unsafe"|"Safe">,
  "category": str<"O1: Hate, Humiliation, Harassment"|"O2: Violence, Harm, or Cruelty"|"O3: Sexual Content"|"O4: Nudity Content"|"O5: Criminal Planning"|"O6: Weapons or Substance Abuse"|"O7: Self-Harm"|"O8: Animal Cruelty"|"O9: Disasters or Emergencies"|"NA: None applying">,
  "rationale": str,
}
"""
backend = RuntimeEndpoint(f"http://localhost:10000")
sgl.set_default_backend(backend)


images = list(Path(base_dir).glob('**/*.jpg'))

# for im_path in tqdm(iamges):
#   out = guard_gen.run(image_path=str(im_path), prompt=prompt)
#   print(out['json_output'])

dataset = []
dataset_name = "unsafe__LlavaGuard_7B_images__blip_laion_cc_sbu_558k"

images.sort(key=lambda x: str(x))

# batch_size = 128
# batch_size = 8
batch_size = 1

for i in tqdm(range(0, len(images), batch_size), total=len(images) // batch_size):
  if i < start_id: continue
  
  batch = images[i:i+batch_size]
  reqs = []
  for img in batch:
    reqs.append({"image_path": str(img), "prompt": prompt})
  
  outputs = guard_gen.run_batch(reqs)
  for img, output in zip(batch, outputs):
    json_output = output["json_output"]
    json_end = '"\n}'
    if not json_output.endswith(json_end):
      json_output += json_end
      print("Incomplete JSON.")
    
    try:
      json_output = eval(json_output)
    except:
      json_output = {"rating" : "NA"}
          
    try:
      rating = json_output['rating']
    except:
      json_output['rating'] = "NA"
      
    imd_id = img.stem
    print(imd_id, rating, end=" || ")
    if rating.lower().startswith("safe"): 
      continue
    
    dataset.append(json_output)
    dataset[-1]["id"] = imd_id
    
    if len(dataset) % 2 == 0:
      # pd.DataFrame(dataset).to_json(f'{output_fname}.json')
      pd.DataFrame(dataset).to_csv(f'{dataset_name}.csv')
  
  # import ipdb; ipdb.set_trace()
  # print(print(outputs[-1]['json_output']))
  # break
  with open('progress.txt', 'a') as f:
    f.write(str(i)+"\n")
  


pd.DataFrame(dataset).to_csv(f'{dataset_name}.csv')