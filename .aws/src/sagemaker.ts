import {Construct} from "constructs";
import {SagemakerEndpointConfiguration} from "@cdktf/provider-aws/lib/sagemaker-endpoint-configuration";
import { config } from './config';
import {SagemakerEndpoint} from "@cdktf/provider-aws/lib/sagemaker-endpoint";
import {SagemakerModel} from "@cdktf/provider-aws/lib/sagemaker-model";
import {DataAwsIamPolicyDocument} from "@cdktf/provider-aws/lib/data-aws-iam-policy-document";
import {IamRole} from "@cdktf/provider-aws/lib/iam-role";
import {PocketVPC} from "@pocket-tools/terraform-modules";
import {IamPolicyAttachment} from "@cdktf/provider-aws/lib/iam-policy-attachment";


export class Sagemaker extends Construct {
  public readonly llmEndpoint: ApplicationSagemaker;
  public readonly embeddingsEndpoint: ApplicationSagemaker;

  constructor(scope: Construct, name: string, vpc: PocketVPC) {
    super(scope, name);

    this.llmEndpoint = new ApplicationSagemaker(this, `llm`, {
        // https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
        modelId: 'mistralai/Mistral-7B-Instruct-v0.1',
        // https://github.com/awslabs/llm-hosting-container/tree/24e89fc612ce97b82b554d92d48ec066bd44dc3e/huggingface/pytorch/tgi/docker/1.1.0
        // https://huggingface.co/blog/sagemaker-huggingface-llm
        modelImage: '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.0.1-tgi1.1.0-gpu-py39-cu118-ubuntu20.04',
        // The model we are using requires a GPU ml.g4dn.xlarge is the cheapest instance with a gpu
        instanceType: 'ml.g5.2xlarge',
        instanceCount: 1,
        vpc,
        modelName: 'llm'
    });

   this.embeddingsEndpoint = new ApplicationSagemaker(this, `embeddings`, {
       // https://huggingface.co/BAAI/bge-large-en-v1.5
        modelId: 'BAAI/bge-large-en-v1.5',
        // https://github.com/awslabs/llm-hosting-container/tree/24e89fc612ce97b82b554d92d48ec066bd44dc3e/huggingface/pytorch/tgi/docker/1.1.0
        // https://huggingface.co/blog/sagemaker-huggingface-llm
        modelImage: '763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-tgi-inference:2.0.1-tgi1.1.0-gpu-py39-cu118-ubuntu20.04',
        instanceType: 'ml.g4dn.xlarge',
        instanceCount: 1,
        vpc,
        modelName: 'embeddings'
   });
  }
}

class ApplicationSagemaker extends Construct {

    public readonly configuration: SagemakerEndpointConfiguration
    public readonly model: SagemakerModel
    public readonly endpoint: SagemakerEndpoint

  constructor(scope: Construct, name: string, options: {vpc: PocketVPC, modelId: string, modelImage: string, modelName: string, instanceType: string, instanceCount: number}) {
    super(scope, name);
    this.model = this.createModel({vpc: options.vpc, modelId: options.modelId, modelImage: options.modelImage, modelName: options.modelName})
    this.configuration = this.createConfiguration({model: this.model, modelName: options.modelName, instanceType: options.instanceType, instanceCount: options.instanceCount})
    this.endpoint = this.createEndpoint({configuration: this.configuration, modelName: options.modelName})
  }


  private createModel(options: {vpc: PocketVPC, modelId: string, modelImage: string, modelName: string}): SagemakerModel {
        const role = new IamRole(this, `model_role`, {
            name: `${config.shortName}-${config.environment}-${options.modelName}-ExecutionRole`,
            assumeRolePolicy: new DataAwsIamPolicyDocument(this, `model_assume_role_policy`, {
                statement: [{
                    actions: ["sts:AssumeRole"],
                    principals: [
                        {
                            type: "Service",
                            identifiers: ["sagemaker.amazonaws.com"]
                        }
                    ]
                }]
                }
            ).json ,
            managedPolicyArns: ['arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'],
            tags: config.tags
        })

        return new SagemakerModel(this, `model`, {
            name: `${config.shortName}-${config.environment}-${options.modelName}`,
            primaryContainer: {
                //https://github.com/aws/deep-learning-containers/blob/master/available_images.md#huggingface-text-generation-inference-containers
                //https://huggingface.co/blog/sagemaker-huggingface-llm
                image: options.modelImage,
                mode: "SingleModel",
                environment: {
                    "HF_MODEL_ID": options.modelId,
                    "HF_TASK": "text-generation",
                    // "HF_HUB_ENABLE_HF_TRANSFER": "false",
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_REGION": options.vpc.region,
                    'SM_NUM_GPUS': '1', //Number of GPU used per replica
                    'MAX_INPUT_LENGTH': '512',//  # Max length of input text
                    // 'MAX_TOTAL_TOKENS': '2048'
                }
            },
            executionRoleArn: role.arn,
            tags: config.tags,
            vpcConfig: {
                subnets: options.vpc.privateSubnetIds,
                securityGroupIds: options.vpc.internalSecurityGroups.ids
            },
            dependsOn: [role]
        })
    }

     private createConfiguration(options: { model: SagemakerModel, modelName: string, instanceType: string, instanceCount: number }): SagemakerEndpointConfiguration {
      return new SagemakerEndpointConfiguration(this, `config`, {
            name: `${config.shortName}-${config.environment}-${options.modelName}`,
            productionVariants: [{
                variantName: 'variant-1',
                modelName: options.model.name,
                // Because we are currently using HuggingFace which downloads the model on startup (and needs a startup healthcheck)
                // and because we use a VPC config we can not use Serverless.
                // In the future when we build our own image, we could probably use serverless.
                containerStartupHealthCheckTimeoutInSeconds: 500,
                initialInstanceCount: options.instanceCount,
                instanceType: options.instanceType,
            }],
            tags: config.tags,
            dependsOn: [options.model]
      })
    }

    private createEndpoint(options: {configuration: SagemakerEndpointConfiguration, modelName: string}): SagemakerEndpoint {
        return new SagemakerEndpoint(this, `endpoint`, {
            endpointConfigName: options.configuration.name,
            name: `${config.shortName}-${config.environment}-${options.modelName}`,
            deploymentConfig: {
                blueGreenUpdatePolicy: {
                    maximumExecutionTimeoutInSeconds: 600,
                    trafficRoutingConfiguration: {
                        type: "ALL_AT_ONCE",
                        waitIntervalInSeconds: 300
                    }
                }
            },
            tags: config.tags,
            dependsOn: [options.configuration]
        })
    }
}