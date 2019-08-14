## 


```sbatch --array=1-10 --export=input="cds3/expt-inputs.txt" cds3/ensemble-run-gpu```

Traceback (most recent call last):
  File "sampnn.py", line 371, in <module>
    main()
  File "sampnn.py", line 99, in main
    args.ensemble, orig_atom_fea_len)
  File "sampnn.py", line 164, in ensemble
    test_ensemble(model_dir, fold_id, ensemble_folds, test_set, fea_len)
  File "sampnn.py", line 325, in test_ensemble
    task="test")
  File "/home/reag2/sampnn/sampnn/utils.py", line 48, in evaluate
    target_norm = normalizer.norm(target)
  File "/home/reag2/sampnn/sampnn/data.py", line 355, in norm
    return (tensor - self.mean) / self.std
RuntimeError: expected device cpu and dtype Float but got device cuda:0 and dtype Float
slurmstepd: error: task_p_post_term: rmdir(/sys/fs/cgroup/cpuset/slurm14672371/slurm14672371.4294967294_0) failed Device or resource busy