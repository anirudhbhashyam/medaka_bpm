import argparse
import os
import logging
from multiprocessing import cpu_count

LOGGER = logging.getLogger(__name__)

def config_logger(logfile_path, logfile_name="medaka_outdir.log"):
    os.makedirs(logfile_path, exist_ok=True)
    
    # Global logger settings
    logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(os.path.join(logfile_path, logfile_name))
                        ])

#TODO write extended help messages, store as string and pass to add_argument() help parameter
def parse_arguments():
    parser = argparse.ArgumentParser(description='Automated heart rate analysis of Medaka embryo videos')
    # General analysis arguments
    parser.add_argument('-i','--indir',     action="store",         dest='indir',       help='Input directory',                 default=False,  required = True)
    parser.add_argument('-o','--outdir',    action="store",         dest='outdir',      help='Output directory',                default=False,  required = True)
    parser.add_argument('-w','--wells',     action="store",         dest='wells',       help='Restrict analysis to wells',      default='[1-96]',     required = False)
    parser.add_argument('-l','--loops',     action="store",         dest='loops',       help='Restrict analysis to loop',       default=None,   required = False)
    parser.add_argument('-c','--channels',  action="store",         dest='channels',    help='Restrict analysis to channel',    default=None,   required = False)
    parser.add_argument('-f','--fps',       action="store",         dest='fps',         help='Frames per second',               default=0.0,    required = False, type=float)
    parser.add_argument('-p','--threads',   action="store",         dest='threads',     help='Threads to use',                  default=1,      required = False, type=int)
    parser.add_argument('-a','--average',   action="store",         dest='average',     help='average',                         default=0.0,    required = False, type=float)

    # Cluster arguments. Index is hidden argument that is set through bash script to assign wells to cluster instances.
    parser.add_argument('--crop',           action="store_true",    dest='crop',        help='Should crop images',                              required = False)
    parser.add_argument('--cluster',        action="store_true",    dest='cluster',     help='Run analysis on a cluster',                       required = False)
    parser.add_argument('-e','--email',     action="store",         dest='email',       help='email for cluster notification',  default='',     required = False)
    parser.add_argument('-m','--maxjobs',   action="store",         dest='maxjobs',     help='maxjobs on the cluster',          default=None,   required = False)
    parser.add_argument('-x', '--lsf_index',action="store",         dest='lsf_index',   help=argparse.SUPPRESS,                                 required = False)
    parser.set_defaults(crop=False, slowmode=False, cluster=False)
    args = parser.parse_args()

    # Adds a trailing slash if it is missing.
    args.indir  = os.path.join(args.indir, '')
    args.outdir = os.path.join(args.outdir, '')

    os.makedirs(args.outdir, exist_ok=True)

    return args

# Processing, done after the logger in the main file has been set up
def process_arguments(args):

    num_cores = cpu_count()
    if (num_cores > args.threads):
        LOGGER.info("the number of virtual processors in your machine is: " + str(num_cores) + " but hou have requested to run on only " + str(args.threads))
        LOGGER.info("You can have faster results (or less errors of the type \"broken pipe\") using the a ideal number of threads in the argument -p in your bash command (E.g.: -p 8). default is 1")

    if not args.maxjobs:
        args.maxjobs = ''
    else:
        args.maxjobs = '%' + args.maxjobs

    if args.channels:
        args.channels   = {c for c in args.channels.split('.')}
    if args.loops:
        args.loops      = {l for l in args.loops.split('.')}

    return args.channels, args.loops