import argparse

def str_to_bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
	else: raise argparse.ArgumentTypeError('Expecting bool, provide either yes/no, true/false, t/f, y/n, 1/0.')

def get_default_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--num_epochs', type=int, default=250)
	parser.add_argument('--num_samples', type=int, default=64) # number of samples and ..
	parser.add_argument('--num_every_nth_epoch', type=int, default=1) # .. how often to store them
	parser.add_argument('--num_gpus', type=int, default=1) # unused
	parser.add_argument('--num_workers', type=int, default=0)

	parser.add_argument("--ckpt", type=str_to_bool, default=False)
	parser.add_argument('--ckpt_path', type=str, default="ckpt")
	parser.add_argument('--data_path', type=str, default="data")
	parser.add_argument('--img_path', type=str, default="img")

	parser.add_argument('--h_dim', type=int, default=128)
	parser.add_argument('--z_dim', type=int, default=100)
	parser.add_argument('--lr_adam', type=float, default=1.e-3)
	parser.add_argument('--beta_1', type=float, default=.5)
	parser.add_argument('--beta_2', type=float, default=.999)
	parser.add_argument('--std', type=float, default=.02)

	return parser
