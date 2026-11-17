"""Set the GitHub repository description, website, and topics after gh auth login."""
import os
import subprocess

REPOSITORY = 'alalbiol/brazil-forest-to-agriculture-benchmark'
DESCRIPTION = ('Temporal Sentinel-1/2 benchmark for forest-to-agriculture screening in Pará and '
               'Mato Grosso, Brazil: references, cached TerraFM features, models and code.')
HOMEPAGE = 'https://www.eotdl.com/datasets/brazil-forest-to-agriculture-benchmark'
TOPICS = ['deforestation', 'land-use-change', 'sentinel-2', 'sentinel-1', 'foundation-models',
          'terrafm', 'eudr', 'amazon', 'remote-sensing', 'reproducible-research']


def main():
    env = {k: v for k, v in os.environ.items() if k not in ('GH_TOKEN', 'GITHUB_TOKEN', 'GH_DEBUG')}
    subprocess.run(['gh', 'auth', 'status', '--hostname', 'github.com'], env=env, check=True)
    command = ['gh', 'repo', 'edit', REPOSITORY, '--description', DESCRIPTION, '--homepage', HOMEPAGE]
    for topic in TOPICS:
        command.extend(['--add-topic', topic])
    subprocess.run(command, env=env, check=True)
    print(f'Updated About fields for {REPOSITORY}.')


if __name__ == '__main__':
    main()
