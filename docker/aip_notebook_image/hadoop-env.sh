# Set Hadoop-specific environment variables here. A

# The only required environment variable is JAVA_HOME. All others are
# optional. When running a distributed configuration it is best to
# set JAVA_HOME in this file, so that it is correctly defined on
# remote nodes.

# The java implementation to use. Required.
#export JAVA_HOME=/etc/alternatives/jre
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export HADOOP_HOME_WARN_SUPPRESS=1

# Hadoop home directory
export HADOOP_HOME=${HADOOP_HOME:-/usr/share/hadoop-client}
export HADOOP_MAPRED_HOME=$HADOOP_HOME
export HADOOP_COMMON_HOME=$HADOOP_HOME
export HADOOP_HDFS_HOME=$HADOOP_HOME
export YARN_HOME=$HADOOP_HOME

# Hadoop Configuration Directory
export HADOOP_CONF_DIR=${HADOOP_CONF_DIR:-/etc/hadoop-client}


# Path to jsvc required by secure HDP 2.0 datanode
export JSVC_HOME=/usr/bin/jsvc


# The maximum amount of heap to use, in MB. Default is 1000.
export HADOOP_HEAPSIZE="64000"

export HADOOP_NAMENODE_INIT_HEAPSIZE="-Xms262144m"

# Extra Java runtime options. Empty by default.
export HADOOP_OPTS="-Djava.net.preferIPv4Stack=true ${HADOOP_OPTS}"

USER="$(whoami)"

# Command specific options appended to HADOOP_OPTS when specified
HADOOP_JOBTRACKER_OPTS="-server -XX:ParallelGCThreads=8 -XX:+UseConcMarkSweepGC
-XX:ErrorFile=/var/log/hadoop/$USER/hs_err_pid%p.log -XX:NewSize=200m
-XX:MaxNewSize=200m -Xloggc:/var/log/hadoop/$USER/gc.log-`date +'%Y%m%d%H%M'`
-verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:+PrintGCDateStamps -Xmx1024m
-Dhadoop.security.logger=INFO,DRFAS -Dmapred.audit.logger=INFO,MRAUDIT
-Dhadoop.mapreduce.jobsummary.logger=INFO,JSA ${HADOOP_JOBTRACKER_OPTS}"

HADOOP_TASKTRACKER_OPTS="-server -Xmx1024m -Dhadoop.security.logger=ERROR,console
-Dmapred.audit.logger=ERROR,console ${HADOOP_TASKTRACKER_OPTS}"


SHARED_HADOOP_NAMENODE_OPTS="-server -XX:ParallelGCThreads=8 -XX:+UseConcMarkSweepGC
-XX:ErrorFile=/var/log/hadoop/$USER/hs_err_pid%p.log -XX:NewSize=24576m
-XX:MaxNewSize=24576m -Xloggc:/var/log/hadoop/$USER/gc.log-`date +'%Y%m%d%H%M'`
-verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:+PrintGCDateStamps
-XX:CMSInitiatingOccupancyFraction=70 -XX:+UseCMSInitiatingOccupancyOnly -Xms262144m
-Xmx262144m -Dhadoop.security.logger=INFO,DRFAS -Dhdfs.audit.logger=INFO,RFAAUDIT"
export HADOOP_NAMENODE_OPTS="${SHARED_HADOOP_NAMENODE_OPTS}
-Dorg.mortbay.jetty.Request.maxFormContentSize=-1 ${HADOOP_NAMENODE_OPTS}"
export HADOOP_DATANODE_OPTS="-server -XX:ParallelGCThreads=4 -XX:+UseConcMarkSweepGC
-XX:ErrorFile=/var/log/hadoop/$USER/hs_err_pid%p.log -XX:NewSize=200m -XX:MaxNewSize=200m
-Xloggc:/var/log/hadoop/$USER/gc.log-`date +'%Y%m%d%H%M'` -verbose:gc -XX:+PrintGCDetails
-XX:+PrintGCTimeStamps -XX:+PrintGCDateStamps -Xms1024m -Xmx1024m
-Dhadoop.security.logger=INFO,DRFAS -Dhdfs.audit.logger=INFO,RFAAUDIT ${HADOOP_DATANODE_OPTS}
-XX:CMSInitiatingOccupancyFraction=70 -XX:+UseCMSInitiatingOccupancyOnly"

export HADOOP_SECONDARYNAMENODE_OPTS="${SHARED_HADOOP_NAMENODE_OPTS} ${HADOOP_SECONDARYNAMENODE_OPTS}"

# The following applies to multiple commands (fs, dfs, fsck, distcp etc)
export HADOOP_CLIENT_OPTS="-Xmx512m $HADOOP_CLIENT_OPTS"


# Command specific options appended to HADOOP_OPTS when specified
export HADOOP_NAMENODE_OPTS="-javaagent:/jolokia-jvm-1.6.0-agent.jar=port=,host=localhost -Dhadoop.security.logger=${HADOOP_SECURITY_LOGGER:-INFO,RFAS} -Dhdfs.audit.logger=${HDFS_AUDIT_LOGGER:-INFO,NullAppender} -server -XX:ParallelGCThreads=15 -XX:+UseParNewGC -XX:+UseConcMarkSweepGC -XX:+ScavengeBeforeFullGC -XX:+CMSScavengeBeforeRemark -XX:CMSInitiatingOccupancyFraction=75 -XX:ErrorFile=/var/log/hadoop/$USER/hs_err_pid%p.log -Xms180g -Xmx180g -Xmn24g -XX:SurvivorRatio=8 -XX:MetaspaceSize=256m -XX:MaxMetaspaceSize=256m -Xloggc:/var/log/hadoop/$USER/gc.log-`date +'%Y%m%d%H%M'` -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:+PrintGCDateStamps -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/var/log/hadoop/$USER/heapdump.hprof -XX:+PrintCommandLineFlags -XX:+PrintTenuringDistribution -XX:+PrintGCApplicationStoppedTime $HADOOP_NAMENODE_OPTS"
export HADOOP_DATANODE_OPTS="-javaagent:/jolokia-jvm-1.6.0-agent.jar=port=,host=localhost -Dhadoop.security.logger=ERROR,RFAS -server -XX:ParallelGCThreads=15 -XX:+UseParNewGC -XX:+UseConcMarkSweepGC -XX:+ScavengeBeforeFullGC -XX:+CMSScavengeBeforeRemark -XX:CMSInitiatingOccupancyFraction=70 -Xms64g -Xmx64g -Xmn8g -XX:SurvivorRatio=8 -XX:MetaspaceSize=256m -XX:MaxMetaspaceSize=256m -Xloggc:/var/log/hadoop/$USER/dn-gc.log-`date +'%Y%m%d%H%M'` -verbose:gc -XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:+PrintGCDateStamps -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/var/log/hadoop/$USER/dn-heapdump.hprof -XX:+PrintCommandLineFlags -XX:+PrintTenuringDistribution -XX:+PrintGCApplicationStoppedTime "
export HADOOP_SECONDARYNAMENODE_OPTS="-Dhadoop.security.logger=${HADOOP_SECURITY_LOGGER:-INFO,RFAS} -Dhdfs.audit.logger=${HDFS_AUDIT_LOGGER:-INFO,NullAppender} $HADOOP_SECONDARYNAMENODE_OPTS"
export HADOOP_PORTMAP_OPTS="-Xmx512m $HADOOP_PORTMAP_OPTS"
export LD_LIBRARY_PATH=/usr/share/hadoop-client/lib/native:/usr/share/java/hadoop/native
export HADOOP_DFSROUTER_OPTS="-javaagent:/jolokia-jvm-1.6.0-agent.jar=port=,host=localhost -Xms64g -Xmx64g -Xmn24g -XX:ParallelGCThreads=15 -XX:+UseParNewGC -XX:+UseConcMarkSweepGC -XX:+ScavengeBeforeFullGC -XX:+CMSScavengeBeforeRemark -XX:CMSInitiatingOccupancyFraction=75 -XX:SurvivorRatio=8 -XX:MetaspaceSize=256m -XX:MaxMetaspaceSize=256m"

HADOOP_NFS3_OPTS="-Xmx1024m -Dhadoop.security.logger=ERROR,DRFAS ${HADOOP_NFS3_OPTS}"
HADOOP_BALANCER_OPTS="-server -Xmx64000m ${HADOOP_BALANCER_OPTS}"


# On secure datanodes, user to run the datanode as after dropping privileges
export HADOOP_SECURE_DN_USER=${HADOOP_SECURE_DN_USER:-""}

# Extra ssh options. Empty by default.
export HADOOP_SSH_OPTS="-o ConnectTimeout=5 -o SendEnv=HADOOP_CONF_DIR"

# Where log files are stored. $HADOOP_HOME/logs by default.
export HADOOP_LOG_DIR=/var/log/hadoop/$USER

# History server logs
export HADOOP_MAPRED_LOG_DIR=/var/log/hadoop-mapreduce/$USER

# Where log files are stored in the secure data environment.
export HADOOP_SECURE_DN_LOG_DIR=/var/log/hadoop/$HADOOP_SECURE_DN_USER

# File naming remote slave hosts. $HADOOP_HOME/conf/slaves by default.
# export HADOOP_SLAVES=${HADOOP_HOME}/conf/slaves

# host:path where hadoop code should be rsync'd from. Unset by default.
# export HADOOP_MASTER=master:/home/$USER/src/hadoop

# Seconds to sleep between slave commands. Unset by default. This
# can be useful in large clusters, where, e.g., slave rsyncs can
# otherwise arrive faster than the master can service them.
# export HADOOP_SLAVE_SLEEP=0.1

# The directory where pid files are stored. /tmp by default.
export HADOOP_PID_DIR=/var/run/hadoop/$USER
export HADOOP_SECURE_DN_PID_DIR=/var/run/hadoop/$HADOOP_SECURE_DN_USER

# History server pid
export HADOOP_MAPRED_PID_DIR=/var/run/hadoop-mapreduce/$USER

YARN_RESOURCEMANAGER_OPTS="-Dyarn.server.resourcemanager.appsummary.logger=INFO,RMSUMMARY"

# A string representing this instance of hadoop. $USER by default.
export HADOOP_IDENT_STRING=$USER

# The scheduling priority for daemon processes. See 'man nice'.

# export HADOOP_NICENESS=10
# Extra Java CLASSPATH elements.  Automatically insert capacity-scheduler.
for f in $HADOOP_HOME/contrib/capacity-scheduler/*.jar; do
    if [ "$HADOOP_CLASSPATH" ]; then
        export HADOOP_CLASSPATH=$HADOOP_CLASSPATH:$f
    else
        export HADOOP_CLASSPATH=$f
    fi
done

# Add database libraries
JAVA_JDBC_LIBS=""
if [ -d "/usr/share/java" ]; then
for jarFile in `ls /usr/share/java | grep -E "(mysql|ojdbc|postgresql|sqljdbc)" 2>/dev/null`
do
JAVA_JDBC_LIBS=${JAVA_JDBC_LIBS}:/usr/share/java/$jarFile
done
fi


#add share_jars
HADOOP_LIBS=""
if [ -d "/usr/share/java/hadoop" ]; then
for jarFile in `ls /usr/share/java/hadoop`
do
HADOOP_LIBS=${HADOOP_LIBS}:/usr/share/java/hadoop/$jarFile
done
fi


# Add libraries to the hadoop classpath - some may not need a colon as they already include it
export HADOOP_CLASSPATH=$HADOOP_CLASSPATH:$HADOOP_CONF_DIR:$HADOOP_HOME/*:$HADOOP_HOME/lib/*:$HADOOP_HOME/lib/native/*:$HADOOP_HOME/share/hadoop/common/*:$HADOOP_HOME/share/hadoop/common/lib/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/hdfs/lib/*:$HADOOP_HOME/share/hadoop/yarn/*:$HADOOP_HOME/share/hadoop/yarn/lib/*:$HADOOP_HOME/share/hadoop/mapreduce/*:$HADOOP_HOME/share/hadoop/mapreduce/lib/*
export HADOOP_CLASSPATH=${JAVA_JDBC_LIBS}:${HADOOP_LIBS}:${HADOOP_CLASSPATH}


# Setting path to hdfs command line
export HADOOP_LIBEXEC_DIR=/usr/share/hadoop-client/libexec

# Mostly required for hadoop 2.0
export JAVA_LIBRARY_PATH=${JAVA_LIBRARY_PATH}:/usr/share/hadoop-client/lib/native:/usr/share/java/hadoop/native



# Enable ACLs on zookeper znodes if required
