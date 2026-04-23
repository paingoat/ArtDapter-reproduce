import os
import sys
from pathlib import Path
from random import choice
from contextlib import contextmanager

CUR_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
PROJECT_ROOT = CUR_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import torch
import einops
import streamlit as st
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything

from ldm.util import instantiate_from_config
from utils import load_weights, resolve_device
from ldm.models.diffusion.custom_ddim import CustomDDIMSampler
from ldm.models.diffusion.ctf_ddim import CTFDDIMSampler

POA_PRINCIPLES = ['balance', 'harmony', 'variety', 'unity', 'contrast', 'emphasis', 'proportion', 'movement', 'rhythm', 'pattern']

DEFAULT_COND_EXAMPLES = [
	{
		'caption': 'A futuristic city skyline at sunset with neon reflections on wet streets.',
		'art_style': 'Expressionism',
		'PoA': [
			'Asymmetric skyline masses are balanced by bright reflections in the foreground.',
			'Neon palette and repeating glow unify the whole composition.',
			'Contrasting light sources and architecture create visual variety.',
			'Consistent perspective and color grading keep all elements coherent.',
			'Warm sunset tones contrast against cool neon blues and violets.',
			'Central boulevard and brightest signage draw immediate focus.',
			'Scale differences between towers, vehicles, and people feel intentional.',
			'Leading road lines guide the eye deep into the scene.',
			'Repeating windows and lights establish a steady visual rhythm.',
			'Facade grids and reflections form layered geometric patterns.'
		]
	},
	{
		'caption': 'A calm mountain lake surrounded by pine trees under morning mist.',
		'art_style': 'Impressionism',
		'PoA': [
			'The shoreline and mirrored mountains create stable visual balance.',
			'Soft brushwork and close tonal families build harmony.',
			'Mist, trees, and ripples add subtle variety without clutter.',
			'Foreground, middle ground, and background connect into one scene.',
			'Light mist and darker pines provide gentle tonal contrast.',
			'The brightest mist patch near the center acts as focal emphasis.',
			'Natural relative sizes of trees and peaks preserve believable proportion.',
			'Ripple directions and slanted branches suggest quiet movement.',
			'Repeated tree silhouettes and water ripples establish rhythm.',
			'Cloud textures and pine clusters create organic pattern repetition.'
		]
	}
]

@contextmanager
def st_horizontal(container):
	with container:
		st.markdown('<span class="hide-element horizontal-marker"></span>', unsafe_allow_html=True)
		yield


def check_state(key, value):
	if key not in st.session_state:
		return False
	return st.session_state[key] == value


def is_diff_model():
	model_options = st.session_state.get('model_options')
	if not model_options:
		return True
	return ('model' not in st.session_state) or \
		not check_state('model_options_checkpoint',	model_options['checkpoint']) or \
		not check_state('model_options_device',			model_options['device']) or \
		not check_state('model_options_precision',	model_options['precision']) or \
        not check_state('model_options_config',     st.session_state.config.model)


def load_inference_config():
	# Select CTF or regular config based on session state
	config_name = st.session_state.get('config_mode', 'regular')
	if config_name == 'ctf':
		config_path = CUR_DIR / '../configs/ctf_inference_config.yaml'
	else:
		config_path = CUR_DIR / '../configs/inference_config.yaml'
	st.session_state.config = OmegaConf.load(str(config_path))


def load_cond_examples():
	if 'cond_examples' not in st.session_state:
		st.session_state.cond_examples = DEFAULT_COND_EXAMPLES


def load_CSS():
	with (CUR_DIR / 'style.css').open() as f:
		css = f.read()
	return f'<style>{css}</style>'


def init_art_controls(reinit=False):
	cond_examples = st.session_state.cond_examples
	if not cond_examples:
		cond_examples = DEFAULT_COND_EXAMPLES
		st.session_state.cond_examples = cond_examples
	example = choice(cond_examples)
	if reinit or 'prompt_value' not in st.session_state:
		st.session_state['prompt_value'] = example['caption']
	if reinit or 'art_controls_art_style' not in st.session_state:
		st.session_state['art_controls_art_style'] = example['art_style']
	for i,principle in enumerate(POA_PRINCIPLES):
		if reinit or f'art_controls_PoA_{principle}' not in st.session_state:
			st.session_state[f'art_controls_PoA_{principle}'] = example['PoA'][i]


def init_sampling_options(reinit=False):
	if reinit or 'sampling_options_quantity' not in st.session_state:
		st.session_state.sampling_options_quantity = 4	# currently only this needs to be cached as image placeholder needs this


def init_display_options(reinit=False):
	if reinit or 'display_options_columns' not in st.session_state:
		st.session_state.display_options_columns = 2


def init_images(reinit=False):
	if reinit or 'artdapted_outputs' not in st.session_state:
		st.session_state.artdapted_outputs = [str(CUR_DIR/'placeholder.svg')] * st.session_state.sampling_options_quantity
	if reinit or 'baseline_outputs' not in st.session_state:
		st.session_state.baseline_outputs = [str(CUR_DIR/'placeholder.svg')] * st.session_state.sampling_options_quantity


def images_quantity_change():
	init_images(reinit=True)


def clear_prompt():
	st.session_state.prompt_value	= ''


def clear_art_controls():
	for key in st.session_state:
		if key.startswith('art_controls'):
			st.session_state[key] = ''


def randomize_values():
	init_art_controls(reinit=True)


def load_model():
	model_options = st.session_state.get('model_options')
	if not model_options:
		raise RuntimeError("Model options are not initialized. Please choose model options in sidebar.")

	st.session_state.model_options_checkpoint =	model_options['checkpoint']
	st.session_state.model_options_config =			st.session_state.config.model #model_options['model_config']
	st.session_state.model_options_device =			model_options['device']
	st.session_state.model_options_precision =	model_options['precision']

	with status_placeholder, st.spinner('Loading model...'):
		trainer = Trainer(inference_mode=True, accelerator='gpu', devices=[st.session_state.model_options_device], precision=st.session_state.model_options_precision)
		with trainer.init_module():
			device = resolve_device(st.session_state.model_options_device)
			weights = load_weights(st.session_state.model_options_checkpoint, device)
			model = instantiate_from_config(st.session_state.model_options_config).to(device)
			missing, unexpected = model.load_state_dict(weights, strict=False)
			if missing:
				print(f"⚠️ Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
			if unexpected:
				print(f"⚠️ Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
			model.eval()
	st.session_state['model'] = model


def render_model_options(container):
	container.header('Model Options')
	model_options = dict(
		device =				container.selectbox("Cuda device", list(range(torch.cuda.device_count()))),
		precision =			container.selectbox("Precision", ['16-mixed', '16-true', '16', 'bf16', 'bf16-true', 'bf16-mixed', 'transformer-engine-float16', '32-true', '32', '64-true', '64', 'transformer-engine']),
		checkpoint =		container.selectbox("Checkpoint", sorted([str(c) for c in CUR_DIR.glob('../ckpt/trained/*.ckpt')], reverse=True))
	)
	st.session_state['model_options'] = model_options
	return model_options


def render_images(container):
	cols = container.columns(st.session_state.display_options_columns)
	for i,img in enumerate(st.session_state.artdapted_outputs):
		cols[i%st.session_state.display_options_columns].image(img)


def render_prompt_controls(container):
	prompt_controls = dict()
	container.markdown('**Prompt**')
	prompt_controls['prompt'] =	container.text_area("Prompt", st.session_state.prompt_value, label_visibility='collapsed')
	container.button('🧹 Clear', key='clear_prompt', on_click=clear_prompt)
	return prompt_controls


def render_art_controls(container):
	art_controls = dict()
	container.markdown('**Art Style**')
	art_styles = ['','Post-Impressionism', 'Expressionism', 'Impressionism',
				'Northern Renaissance', 'Realism', 'Romanticism', 'Symbolism', 'Art Nouveau (Modern)', 'Naïve Art (Primitivism)',
				'Baroque', 'Rococo', 'Abstract Expressionism', 'Cubism', 'Color Field Painting', 'Pop Art', 'Pointillism',
				'Early Renaissance', 'Ukiyo-e', 'Mannerism (Late Renaissance)', 'High Renaissance', 'Fauvism', 'Minimalism',
				'Action painting', 'Contemporary Realism', 'Synthetic Cubism', 'New Realism', 'Analytical Cubism']

	col1, _, _ = container.columns(3)
	art_controls['art_style'] =	col1.selectbox('Art style', art_styles, index=art_styles.index(st.session_state.art_controls_art_style),
				placeholder='Choose an art-style or none at all.', label_visibility='collapsed')

	container.markdown('**Principles of Art**')
	col1, col2 = container.columns(2)
	art_controls['PoA_balance'] =			col1.text_area('Balance',			value=st.session_state.art_controls_PoA_balance)
	art_controls['PoA_harmony'] =			col1.text_area('Harmony',			value=st.session_state.art_controls_PoA_harmony)
	art_controls['PoA_variety'] =			col1.text_area('Variety',			value=st.session_state.art_controls_PoA_variety)
	art_controls['PoA_unity'] =				col1.text_area('Unity',				value=st.session_state.art_controls_PoA_unity)
	art_controls['PoA_contrast'] =		col1.text_area('Contrast',		value=st.session_state.art_controls_PoA_contrast)
	art_controls['PoA_emphasis'] =		col2.text_area('Emphasis',		value=st.session_state.art_controls_PoA_emphasis)
	art_controls['PoA_proportion'] =	col2.text_area('Proportion',	value=st.session_state.art_controls_PoA_proportion)
	art_controls['PoA_movement'] =		col2.text_area('Movement',		value=st.session_state.art_controls_PoA_movement)
	art_controls['PoA_rhythm'] =			col2.text_area('Rhythm',			value=st.session_state.art_controls_PoA_rhythm)
	art_controls['PoA_pattern'] =			col2.text_area('Pattern',			value=st.session_state.art_controls_PoA_pattern)

	container.button('🧹 Clear art controls', on_click=clear_art_controls)
	return art_controls


def render_sampling_options(container):
	col1, col2, col3 = container.columns(3)

	sampling_options = dict(
		seed =				col1.number_input('Seed',				value=42,		min_value=-1,		max_value=2147483647, step=1),
		quantity =		col1.slider('Outputs',											min_value=1,		max_value=12, 				step=1, key='sampling_options_quantity', on_change=images_quantity_change),
		resolution = 	col1.slider('Resolution',				value=512,	min_value=256,	max_value=768,				step=64),
		steps =				col2.slider('Diffusion Steps',	value=50,		min_value=1,		max_value=100,				step=1),
		CFG_scale =		col2.slider('Guidance Scale',		value=7.5,	min_value=0.1,	max_value=30.,				step=0.1),
		strategy =		col3.radio('Sampling strategy', ["regular", "ddim"], index=1, horizontal=True)
	)
	if sampling_options['strategy'] == 'ddim':
		sampling_options['ddim_eta'] = col3.number_input("η (DDIM)", value=0.)

	# CTF controls (Only visible in CTF mode)
	if st.session_state.get('config_mode') == 'ctf':
		ctf_defaults = st.session_state.config.get('ctf_sampling', {}) if hasattr(st.session_state, 'config') else {}
		default_layout      = float(ctf_defaults.get('layout_end',        0.30))
		default_content     = float(ctf_defaults.get('content_end',       0.65))
		default_blend       = float(ctf_defaults.get('blend_window',      0.08))
		default_preserve    = float(ctf_defaults.get('preserve_strength', 0.30))

		st.markdown('**CTF Phase Thresholds (Temporal Proxy Prompt)**')
		sampling_options['layout_end'] = st.slider(
			'Phase 1: Layout End', value=default_layout, min_value=0.0, max_value=0.5, step=0.05,
			help='Fraction of steps where Phase 1 (CLIP Layout) ends.')
		sampling_options['content_end'] = st.slider(
			'Phase 2: Content End', value=default_content, min_value=0.2, max_value=0.9, step=0.05,
			help='Fraction of steps where Phase 2 (CLIP Content) ends. Phase 3 (Style via ArtDapter) runs after.')
		sampling_options['blend_window'] = st.slider(
			'Blend Window (half-width)', value=default_blend, min_value=0.0, max_value=0.20, step=0.01,
			help='Cosine cross-fade window around Content End. In window, both content and style branches run (~2x cost).')
		sampling_options['preserve_strength'] = st.slider(
			'Preserve Content Anchor', value=default_preserve, min_value=0.0, max_value=0.60, step=0.05,
			help='SDEdit-lite pull toward the Phase-2 content latent during Phase 3. Higher = layout stays more fixed.')
		sampling_options['show_stages'] = st.checkbox(
			'Show 3-Stage Progression', value=True,
			help='Extract and display intermediate layout/content/style stages.')
	else:
		sampling_options['layout_end']        = 0.30
		sampling_options['content_end']       = 0.65
		sampling_options['blend_window']      = 0.08
		sampling_options['preserve_strength'] = 0.30
		sampling_options['show_stages']       = False

	return sampling_options


def render_display_options(container):
	col, _, _ = container.columns(3)
	col.number_input('Columns', min_value=1, step=1, key='display_options_columns', on_change=images_quantity_change)


def decode_to_numpy(z, model):
	"""Decode latents to numpy images list."""
	x = model.decode_first_stage(z)
	x = (einops.rearrange(x, 'b c h w -> b h w c') * 0.5 + 0.5).clamp(0, 1)
	return [img.cpu().numpy() for img in x]


@torch.no_grad()
def generate():
	if is_diff_model():
		load_model()

	seed_everything(sampling_options['seed'])

	# Aliases
	model =				st.session_state['model']
	prompt =			prompt_controls['prompt']
	art_style = art_controls['art_style']
	PoA = [art_controls['PoA_balance'], art_controls['PoA_harmony'], art_controls['PoA_variety'],
						 art_controls['PoA_unity'],	art_controls['PoA_contrast'], art_controls['PoA_emphasis'],
						 art_controls['PoA_proportion'], art_controls['PoA_movement'], art_controls['PoA_rhythm'],
						 art_controls['PoA_pattern']]
	sample_quantity = sampling_options['quantity']
	sample_resolution = sampling_options['resolution']
	sampling_steps = sampling_options['steps']
	ddim_eta = sampling_options.get('ddim_eta', 0.0)
	cfg_scale = sampling_options['CFG_scale']
	style_start = sampling_options.get('style_start', 0.7)
	show_stages = sampling_options.get('show_stages', True)

		# Check if model is CTF-capable
	is_ctf = hasattr(model, 'get_ctf_conditioning')

	if is_ctf:
		# ── CTF Pipeline ─────────────────────────────────────────

		# Step 1: Decompose prompt via GPT
		with status_placeholder, st.spinner('🤖 Decomposing prompt via GPT...'):
			cond = model.get_ctf_conditioning(
				captions=[prompt] * sample_quantity,
				art_styles=[art_style] * sample_quantity,
				PoAs=[PoA] * sample_quantity,
				sample_quantity=sample_quantity,
			)
			un_cond = model.get_unconditional_conditioning(sample_quantity)
			# Save decomposed prompts for debug panel
			if hasattr(model, '_last_decomposed'):
				st.session_state['ctf_prompts'] = model._last_decomposed

		ctf_sampler = CTFDDIMSampler(model)

		sample_kwargs = dict(
			S=sampling_steps,
			batch_size=sample_quantity,
			shape=(4, sample_resolution // 8, sample_resolution // 8),
			conditioning=cond,
			unconditional_guidance_scale=cfg_scale,
			unconditional_conditioning=un_cond,
			eta=ddim_eta,
			verbose=False,
			img_callback=None
		)

		# Retrieve exact options from sliders
		layout_end        = sampling_options.get('layout_end',        0.30)
		content_end       = sampling_options.get('content_end',       0.65)
		blend_window      = sampling_options.get('blend_window',      0.08)
		preserve_strength = sampling_options.get('preserve_strength', 0.30)

		with status_placeholder, st.spinner('CTF sampling...'):
			artdapted_z_samples, intermediates = ctf_sampler.sample(
				**sample_kwargs,
				layout_end=layout_end,
				content_end=content_end,
				blend_window=blend_window,
				preserve_strength=preserve_strength,
			)

		if show_stages and 'stage_structure' in intermediates and 'stage_content' in intermediates:
			st.session_state['ctf_stage1_outputs'] = decode_to_numpy(intermediates['stage_structure'], model)
			st.session_state['ctf_stage2_outputs'] = decode_to_numpy(intermediates['stage_content'], model)
		else:
			st.session_state.pop('ctf_stage1_outputs', None)
			st.session_state.pop('ctf_stage2_outputs', None)

	else:
		# ── Original Pipeline (backward compatible) ────────────
		st.session_state.pop('ctf_prompts', None)
		st.session_state.pop('ctf_stage1_outputs', None)
		st.session_state.pop('ctf_stage2_outputs', None)

		caption = model.apply_prompt_template([prompt]* sample_quantity, [art_style]* sample_quantity, [PoA]* sample_quantity)
		cond = dict(c_crossattn =	[model.get_learned_conditioning(caption)] )
		un_cond = dict(c_crossattn=[model.get_unconditional_conditioning(sample_quantity)])

		with status_placeholder, st.spinner('Sampling...'):
			if sampling_options['strategy'] == 'regular':
				kwargs = dict(
					batch_size=sample_quantity,
					unconditional_conditioning=un_cond,
					ddim=False
				)
				artdapted_z_samples, _ = model.sample_log(cond=cond, **kwargs)
			elif sampling_options['strategy'] == 'ddim':
				ddim_sampler = CustomDDIMSampler(model)
				kwargs = dict(
					S=sampling_steps,
					batch_size=sample_quantity,
					shape=(4, sample_resolution // 8, sample_resolution // 8),
					verbose=False,
					eta=ddim_eta,
					unconditional_guidance_scale=cfg_scale,
					unconditional_conditioning=un_cond,
				)
				artdapted_z_samples, _ = ddim_sampler.sample(conditioning=cond, **kwargs)

	st.session_state.artdapted_outputs = decode_to_numpy(artdapted_z_samples, model)
	st.toast(f'Output{"s" if st.session_state.sampling_options_quantity > 1 else ""} generated!', icon='🎉')


def render_ctf_debug_panel(container):
	"""Display CTF Phase Analysis: GPT decomposition and 3-Stage image progression."""
	if 'ctf_prompts' not in st.session_state:
		return

	with container.expander('🔍 CTF Phase Analysis', expanded=True):
		# Prompt decomposition display
		st.markdown('#### 🤖 GPT Prompt Decomposition')
		p = st.session_state['ctf_prompts']
		c1, c2, c3 = st.columns(3)
		c1.info(f"**P1 — Spatial Layout:**\n\n{p.get('prompt1', '')}")
		c2.success(f"**P2 — Content:**\n\n{p.get('prompt2', '')}")
		c3.warning(f"**P3 — Full + Style:**\n\n{p.get('prompt3', '')}")

		# 3-Stage Progression Visualization
		if 'ctf_stage1_outputs' in st.session_state and 'ctf_stage2_outputs' in st.session_state and 'artdapted_outputs' in st.session_state:
			st.markdown('---')
			st.markdown('#### 📈 Generation Progression')
			st.caption('Tiến trình Temporal Proxy Prompt (Pha 1 → Pha 2 → Pha 3)')

			stage_cols = st.columns(3)
			
			with stage_cols[0]:
				st.markdown('**Phase 1: Layout (Grayscale/Blockout)**')
				for img in st.session_state['ctf_stage1_outputs']:
					st.image(img, use_container_width=True)
			
			with stage_cols[1]:
				st.markdown('**Phase 2: Content (Details added)**')
				for img in st.session_state['ctf_stage2_outputs']:
					st.image(img, use_container_width=True)
					
			with stage_cols[2]:
				st.markdown('**Phase 3: Final Art Style**')
				for img in st.session_state['artdapted_outputs']:
					st.image(img, use_container_width=True)


# Preprocess
load_inference_config()
load_cond_examples()
init_art_controls()
init_sampling_options()
init_display_options()
init_images()

# Layout
st.set_page_config(
	page_title =	'ArtDapted Model Inference',
	page_icon =		'🎨',
	layout =			'wide')
st.title('ArtDapted Model Inference')
st.markdown(load_CSS(), unsafe_allow_html=True)
# Pipeline mode selector
pipeline_mode = st.sidebar.radio('Pipeline Mode', ['regular', 'ctf'], index=0, horizontal=True,
									 help='CTF = Coarse-to-Fine prompt orchestration')
if st.session_state.get('config_mode') != pipeline_mode:
	st.session_state['config_mode'] = pipeline_mode
	load_inference_config()

st.button('🎨 **GENERATE**', type='primary', on_click=generate)
model_options = render_model_options(st.sidebar)

status_placeholder = st.empty()
top = st.container()
bot = st.container()

left, right = top.columns(2)
left.markdown('### Controls')
left.button('🪄 Randomize prompt & controls', on_click=randomize_values, use_container_width=True)
prompt_controls = render_prompt_controls(left)
left.divider()
art_controls = render_art_controls(left)

right.markdown('### Outputs')
render_images(right)
tab1, tab2 = right.tabs(['**Sampling Options**', '**Display Options**'])
sampling_options = render_sampling_options(tab1)
render_display_options(tab2)

# CTF Phase Analysis panel (below main layout)
render_ctf_debug_panel(bot)
