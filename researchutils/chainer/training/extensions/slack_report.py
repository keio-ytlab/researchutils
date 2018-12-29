from chainer.training import extension
from chainer.training.extensions import log_report as log_report_module
from slackclient import SlackClient as slackclient_module


class SlackReport(extension.Extension):
    """
    Sends learning status periodically to slack's channel

    Basic behavior is same as chainer.training.extensions.PrintReport
    The main difference is that this extension sends result to specified slack channel
    """

    def __init__(self, token_or_client, entries, channel='general', log_report='LogReport'):
        if isinstance(token_or_client, slackclient_module):
            self._slack_client = token_or_client
        elif isinstance(token_or_client, str):
            self._slack_client = slackclient_module(token=token_or_client)
        else:
            raise TypeError(
                'Given argument is neither SlackClient nor token!!')
        self._entries = entries
        self._channel = channel
        self._log_report = log_report

        # create channel if does not exist
        if not self._join_channel(channel):
            raise ValueError(
                'Could not join to given channel: {}'.format(channel))

        # format information
        entry_widths = [max(10, len(s)) for s in entries]

        self._header = '  '.join(('{:%d}' % w for w in entry_widths)).format(
            *entries) + '\n'

        templates = []
        for entry, w in zip(entries, entry_widths):
            templates.append((entry, '{:<%dg}  ' % w, ' ' * (w + 2)))
        self._templates = templates

    def __call__(self, trainer):
        log_report = self._log_report
        if isinstance(log_report, str):
            log_report = trainer.get_extension(log_report)
        elif isinstance(log_report, log_report_module.LogReport):
            log_report(trainer)  # update the log report
        else:
            raise TypeError('log report has a wrong type %s' %
                            type(log_report))

        message = self._header
        observations = log_report.log
        for observation in observations:
            for entry, template, empty in self._templates:
                if entry in observation:
                    message += template.format(observation[entry])
                else:
                    message += empty
            message += '\n'
        self._send_message(text=message, channel=self._channel)

    def initialize(self, trainer):
        self._send_message(
            text='--------training started--------', channel=self._channel)

    def finalize(self):
        self._send_message(
            text='--------training finished--------', channel=self._channel)

    def serialize(self, serializer):
        log_report = self._log_report
        if isinstance(log_report, log_report_module.LogReport):
            log_report.serialize(serializer['_log_report'])

    def _send_message(self, text, channel):
        if not isinstance(self._slack_client, slackclient_module):
            raise TypeError('slack client has a wrong type %s' %
                            type(self._slack_client))
        try:
            result = self._slack_client.api_call(
                "chat.postMessage",
                channel=channel,
                text=text
            )
            return self._is_success_api_call(result)
        except:
            return False

    def _join_channel(self, channel):
        if not isinstance(self._slack_client, slackclient_module):
            raise TypeError('slack client has a wrong type %s' %
                            type(self._slack_client))
        try:
            result = self._slack_client.api_call(
                "channels.join",
                name=channel
            )
            return self._is_success_api_call(result)
        except:
            # Treat any network error as success to avoid unexpected 
            return False

    def _is_success_api_call(self, result):
        return "ok" in result
