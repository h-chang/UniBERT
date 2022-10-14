from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from textattack import Attacker, CommandLineAttackArgs, DatasetArgs, ModelArgs
from textattack.commands import TextAttackCommand


class AttackCommand(TextAttackCommand):
    """The TextAttack attack module:

    A command line parser to run an attack from user specifications.
    """

    def run(self, args):
        attack_args = CommandLineAttackArgs(**vars(args))
        dataset = DatasetArgs._create_dataset_from_args(attack_args)

        if attack_args.interactive:
            model_wrapper = ModelArgs._create_model_from_args(attack_args)
            attack = CommandLineAttackArgs._create_attack_from_args(
                attack_args, model_wrapper
            )
            Attacker.attack_interactive(attack)
        else:
            model_wrapper = ModelArgs._create_model_from_args(attack_args)
            ## NOTE mod start
            if hasattr(model_wrapper.model,'bert'):
                model_wrapper.model.bert.config.ver = attack_args.ver
                model_wrapper.model.bert.config.margin = attack_args.margin
                model_wrapper.model.bert.config.penalty = attack_args.penalty
            elif hasattr(model_wrapper.model,'roberta'):
                model_wrapper.model.roberta.config.ver = attack_args.ver
                model_wrapper.model.roberta.config.margin = attack_args.margin
                model_wrapper.model.roberta.config.penalty = attack_args.penalty
            ## NOTE mod end
            
            attack = CommandLineAttackArgs._create_attack_from_args(
                attack_args, model_wrapper
            )
            
            attacker = Attacker(attack, dataset, attack_args)
            
            attacker.attack_dataset()
            

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "attack",
            help="run an attack on an NLP model",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser = CommandLineAttackArgs._add_parser_args(parser)
        parser.set_defaults(func=AttackCommand())
