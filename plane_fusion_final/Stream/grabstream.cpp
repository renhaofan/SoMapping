#include "grabstream.h"

#include <QDebug>

GrabStream::GrabStream(QObject *parent)
    : QThread(parent)
{
    // if opencv needed, make sure multithread error
//    qRegisterMetaType<cv::Mat>("cv::Mat");
    qRegisterMetaType<GrabStream::Status>("GrabStream::Status");
}

GrabStream::~GrabStream()
{
    m_runStatus = Status::STOP;
    // quit safely
    this->quit(); // send quit signal
    this->wait(); // wait until quit
    this->deleteLater(); // release cpu memory

}

void GrabStream::setRunning()
{
    m_runStatus = Status::RUNNING;
}

void GrabStream::setStop()
{
    qDebug() << "Enter setStop Function";
    if (this->isRunning())
    {
        m_runStatus = Status::STOP;
        m_frameIndex = 0;
//        this->quit(); // send quit signal
//        this->wait(); // wait until quit
//        this->deleteLater(); // release cpu memory
    }
    qDebug() << "Exit setStop Function";
}

void GrabStream::setPause()
{
    m_runStatus = Status::PAUSE;
}

void GrabStream::setResume()
{
    m_runStatus = Status::RUNNING;
}

void GrabStream::printStatus()
{
    auto tmp_status = static_cast<std::underlying_type<Status>::type>(m_runStatus);
    switch (tmp_status)
    {
        case 0:
            qDebug() << "Status:ERROR";
            break;
        case 1:
            qDebug() << "Status:STOP";
            break;
        case 2:
            qDebug() << "Status:RUNNING";
            break;
        case 3:
            qDebug() << "Status:PAUSE";
            break;
        default:
            Q_ASSERT_X(false, "Error", "Unknown status");
            break;
    }
}

void GrabStream::run()
{
    qDebug() << "enter thread : " << QThread::currentThreadId();


    if (m_fileName.isEmpty())
    {
        qDebug() << "fileName empty";
        return;
    }

    while (true)
    {
        if (m_runStatus == Status::RUNNING)
        {
            m_depthQImage.load(m_fileName + "depth/"+ QString::number(m_frameIndex)+".png");
            m_colorQImage.load(m_fileName + "color/"+ QString::number(m_frameIndex)+".jpg");
            // if grab frame is empty, don't send sign
            if (m_depthQImage.isNull() || m_colorQImage.isNull())
            {
                qDebug() << "stop to grab stream " << QString::number(m_frameIndex);
                Q_ASSERT_X(false, "Error", "Failed to grab frame.");
                continue;;
            }
            m_frameIndex++;
            emit sigSendFrame(m_colorQImage, m_depthQImage);
        }
        else if (m_runStatus == Status::PAUSE)
        {
        }
        else if (m_runStatus == Status::STOP)
        {
            qDebug() << "stop to grab stream";
            break;
        }
        else if (m_runStatus == Status::ERROR)
        {
            //  work when no definiton of macro QT_NO_DEBUG
            printStatus();
            Q_ASSERT_X(false, "Error", "ERROR Status");
        }
    }

    qDebug() << "exit thread : " << QThread::currentThreadId();

}
