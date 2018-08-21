import pytest
import mock
from mock import Mock
from slackclient import SlackClient
from chainer import training
from chainer.training.extensions import log_report as log_report_module
import researchutils.chainer.training.extensions.slack_report as slack_report


class TestSlackReport(object):
    def test_non_slack_client_instance(self):
        invalid_slack_client = 'this is not a slackclient instance'
        channel = 'test channel'
        entries = ['main/loss']
        kwargs = dict(slack_client=invalid_slack_client,
                      entries=entries,
                      channel=channel)

        with pytest.raises(TypeError):
            slack_report.SlackReport(**kwargs)

    def test_failed_joining_channel(self):
        join_channel_fail_client = Mock(spec=SlackClient)
        join_channel_fail_client.api_call.return_value = "ng"
        channel = 'test channel'
        entries = ['main/loss']
        kwargs = dict(slack_client=join_channel_fail_client,
                      entries=entries,
                      channel=channel)

        with pytest.raises(ValueError):
            slack_report.SlackReport(**kwargs)

    def test_send_message(self):
        mock_client = Mock(spec=SlackClient)
        mock_client.api_call.return_value = "ok"
        channel = 'test channel'
        entries = ['main/loss']
        kwargs = dict(slack_client=mock_client,
                      entries=entries,
                      channel=channel)

        mock_trainer = Mock(spec=training.Trainer)
        mock_log_report = Mock(spec=log_report_module.LogReport)
        mock_log_report.log = []
        mock_trainer.get_extension.return_value = mock_log_report

        reporter = slack_report.SlackReport(**kwargs)
        reporter(trainer=mock_trainer)

        mock_client.api_call.assert_called_with(
            "chat.postMessage",
            channel=channel,
            text=mock.ANY)

    def test_send_initialize_message(self):
        mock_client = Mock(spec=SlackClient)
        mock_client.api_call.return_value = "ok"
        channel = 'test channel'
        entries = ['main/loss']
        kwargs = dict(slack_client=mock_client,
                      entries=entries,
                      channel=channel)

        reporter = slack_report.SlackReport(**kwargs)
        reporter.initialize(trainer=None)

        mock_client.api_call.assert_called_with(
            "chat.postMessage",
            channel=channel,
            text=mock.ANY)

    def test_send_finalize_message(self):
        mock_client = Mock(spec=SlackClient)
        mock_client.api_call.return_value = "ok"
        channel = 'test channel'
        entries = ['main/loss']
        kwargs = dict(slack_client=mock_client,
                      entries=entries,
                      channel=channel)

        reporter = slack_report.SlackReport(**kwargs)
        reporter.finalize()

        mock_client.api_call.assert_called_with(
            "chat.postMessage",
            channel=channel,
            text=mock.ANY)


if __name__ == '__main__':
    pytest.main()
